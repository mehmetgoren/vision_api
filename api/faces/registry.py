"""SQLite-backed face registry with on-disk .npy embedding storage.

Each registered face stores one L2-normalised 512-d ArcFace embedding (from
InsightFace's buffalo_l pack). Matching is a single matmul over the
in-memory cache, which is rebuilt lazily after register/delete. Cosine
similarity == dot product on already-normalised vectors.

Layout:
    {data_dir}/faces.db                     — SQLite metadata
    {data_dir}/embeddings/{userid}/{id}.npy — one 512-d float32 per face
"""
from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import threading
import uuid
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _default_data_dir() -> Path:
    return Path(os.environ.get("FACE_REGISTRY_DIR", "data"))


_SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
    userid     TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS face_embeddings(
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    userid          TEXT NOT NULL,
    embedding_path  TEXT NOT NULL,
    quality         REAL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(userid) REFERENCES users(userid) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS ix_face_embeddings_userid ON face_embeddings(userid);
"""


class FaceRegistry:
    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = (data_dir or _default_data_dir()).resolve()
        self.embeddings_dir = self.data_dir / "embeddings"
        self.db_path = self.data_dir / "faces.db"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._cache: dict[str, np.ndarray] | None = None
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    def _invalidate_cache(self) -> None:
        with self._lock:
            self._cache = None

    def _ensure_cache(self) -> dict[str, np.ndarray]:
        with self._lock:
            if self._cache is not None:
                return self._cache

            cache: dict[str, list[np.ndarray]] = {}
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT userid, embedding_path FROM face_embeddings ORDER BY userid, id"
                ).fetchall()

            for userid, emb_path in rows:
                p = Path(emb_path)
                if not p.is_absolute():
                    p = self.data_dir / p
                if not p.exists():
                    logger.warning("Skipping missing embedding %s for %s", p, userid)
                    continue
                try:
                    arr = np.load(p).astype(np.float32, copy=False)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to load %s: %s", p, exc)
                    continue
                cache.setdefault(userid, []).append(arr)

            self._cache = {uid: np.vstack(vs) for uid, vs in cache.items() if vs}
            logger.info(
                "FaceRegistry cache rebuilt: users=%d total_embeddings=%d",
                len(self._cache),
                sum(arr.shape[0] for arr in self._cache.values()),
            )
            return self._cache

    def list_userids(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT userid FROM users ORDER BY userid").fetchall()
        return [r[0] for r in rows]

    def register(
        self,
        userid: str,
        embeddings: list[np.ndarray],
        qualities: list[float] | None = None,
    ) -> int:
        if not embeddings:
            return 0
        qualities = qualities or [0.0] * len(embeddings)
        user_dir = self.embeddings_dir / userid
        user_dir.mkdir(parents=True, exist_ok=True)

        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO users(userid) VALUES (?)",
                (userid,),
            )
            inserted = 0
            for emb, qual in zip(embeddings, qualities):
                if emb.shape != (512,):
                    logger.warning("Skipping non-512-d embedding for %s: shape=%s", userid, emb.shape)
                    continue
                normed = emb.astype(np.float32, copy=False)
                norm = float(np.linalg.norm(normed))
                if norm > 0.0:
                    normed = normed / norm
                fname = f"{uuid.uuid4().hex}.npy"
                fpath = user_dir / fname
                np.save(fpath, normed)
                rel = fpath.relative_to(self.data_dir).as_posix()
                conn.execute(
                    "INSERT INTO face_embeddings(userid, embedding_path, quality) VALUES (?, ?, ?)",
                    (userid, rel, float(qual)),
                )
                inserted += 1
            conn.commit()

        self._invalidate_cache()
        logger.info("Registered %d face(s) for userid=%s", inserted, userid)
        return inserted

    def delete(self, userid: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM users WHERE userid = ?", (userid,))
            existed = cur.rowcount > 0
            conn.commit()

        user_dir = self.embeddings_dir / userid
        if user_dir.exists():
            shutil.rmtree(user_dir, ignore_errors=True)

        if existed:
            self._invalidate_cache()
            logger.info("Deleted userid=%s and its embeddings", userid)
        return existed

    def match(
        self,
        query: np.ndarray,
        threshold: float = 0.4,
    ) -> tuple[str | None, float]:
        """Return (best_userid, similarity). userid is None if no entry passes
        `threshold` or the registry is empty.
        """
        cache = self._ensure_cache()
        if not cache:
            return None, 0.0

        q = query.astype(np.float32, copy=False)
        qn = float(np.linalg.norm(q))
        if qn > 0.0:
            q = q / qn

        best_user: str | None = None
        best_sim: float = -1.0
        for userid, mat in cache.items():
            sims = mat @ q  # (N,) dot products, already normalised on both sides
            top = float(sims.max())
            if top > best_sim:
                best_sim = top
                best_user = userid

        if best_user is None or best_sim < threshold:
            return None, max(best_sim, 0.0)
        return best_user, best_sim


face_registry = FaceRegistry()
