"""CodeProject.AI face endpoints — detect, recognize, register, list, delete, match."""
from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, Request, UploadFile

from api.faces.pipeline import face_pipeline
from api.faces.registry import face_registry
from api.image_input import decode_image_bytes, read_image_upload
from api.response import Timing, build_response, error_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/vision/face", tags=["face"])

MODULE_ID = "FaceProcessing"
MODULE_NAME = "Face Processing"


def _bbox_prediction(emb, label: str, confidence: float, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    x1, y1, x2, y2 = emb.bbox
    pred: dict[str, Any] = {
        "confidence": round(float(confidence), 4),
        "label": label,
        "x_min": int(x1),
        "y_min": int(y1),
        "x_max": int(x2),
        "y_max": int(y2),
    }
    if extra:
        pred.update(extra)
    return pred


@router.post("")
async def face_detect(
    image: Annotated[UploadFile, File(description="Image to analyse")],
    min_confidence: Annotated[float, Form(ge=0.0, le=1.0)] = 0.4,
) -> dict[str, Any]:
    command = "detect"
    with Timing() as timing:
        try:
            img = await read_image_upload(image, field_name="image")
        except Exception as exc:  # noqa: BLE001
            return error_response(str(exc), module_id=MODULE_ID, command=command, timing=timing)

        with timing.inference():
            faces = face_pipeline.analyse(img, min_confidence=min_confidence)

        predictions = [_bbox_prediction(f, "face", f.det_score) for f in faces]

    logger.info("/v1/vision/face: count=%d total_ms=%d", len(predictions), timing.process_ms)
    return build_response(
        predictions,
        module_id=MODULE_ID,
        module_name=MODULE_NAME,
        command=command,
        timing=timing,
    )


@router.post("/recognize")
async def face_recognize(
    image: Annotated[UploadFile, File(description="Image to analyse")],
    min_confidence: Annotated[float, Form(ge=0.0, le=1.0)] = 0.4,
) -> dict[str, Any]:
    command = "recognize"
    with Timing() as timing:
        try:
            img = await read_image_upload(image, field_name="image")
        except Exception as exc:  # noqa: BLE001
            return error_response(str(exc), module_id=MODULE_ID, command=command, timing=timing)

        with timing.inference():
            faces = face_pipeline.analyse(img, min_confidence=0.0)
            predictions: list[dict[str, Any]] = []
            for f in faces:
                userid, similarity = face_registry.match(f.embedding, threshold=min_confidence)
                label = userid or "unknown"
                predictions.append(
                    _bbox_prediction(
                        f,
                        label,
                        similarity if userid else f.det_score,
                        extra={"userid": label},
                    )
                )

    logger.info(
        "/v1/vision/face/recognize: faces=%d matched=%d total_ms=%d",
        len(faces), sum(1 for p in predictions if p["userid"] != "unknown"), timing.process_ms,
    )
    return build_response(
        predictions,
        module_id=MODULE_ID,
        module_name=MODULE_NAME,
        command=command,
        timing=timing,
    )


@router.post("/register")
async def face_register(request: Request) -> dict[str, Any]:
    """CodeProject's contract: `userid` plus one or more `imageN` files
    (image1, image2, ...). FastAPI can't bind a variable file count from
    OpenAPI, so we parse the multipart form manually.
    """
    command = "register"
    with Timing() as timing:
        try:
            form = await request.form()
        except Exception as exc:  # noqa: BLE001
            return error_response(f"Could not parse form: {exc}", module_id=MODULE_ID, command=command, timing=timing)

        userid_raw = form.get("userid")
        if not isinstance(userid_raw, str) or not userid_raw.strip():
            return error_response("'userid' is required", module_id=MODULE_ID, command=command, timing=timing)
        userid = userid_raw.strip()

        upload_keys = sorted(
            (k for k in form.keys() if k.lower().startswith("image") and not isinstance(form[k], str)),
            key=lambda k: (len(k), k),
        )
        if not upload_keys:
            return error_response(
                "At least one imageN file is required (image1, image2, ...)",
                module_id=MODULE_ID,
                command=command,
                timing=timing,
            )

        embeddings = []
        qualities = []
        skipped = 0
        with timing.inference():
            for key in upload_keys:
                upload = form[key]
                if not isinstance(upload, UploadFile):
                    continue
                raw = await upload.read()
                if not raw:
                    skipped += 1
                    continue
                try:
                    img = decode_image_bytes(raw, field_name=key)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("register: %s could not be decoded: %s", key, exc)
                    skipped += 1
                    continue
                faces = face_pipeline.analyse(img, min_confidence=0.0)
                if not faces:
                    logger.info("register: no face in %s for %s", key, userid)
                    skipped += 1
                    continue
                # Pick the highest-confidence face per image (typical enrolment flow).
                best = max(faces, key=lambda f: f.det_score)
                embeddings.append(best.embedding)
                qualities.append(best.det_score)

        if not embeddings:
            return error_response(
                f"No usable faces found in {len(upload_keys)} image(s)",
                module_id=MODULE_ID,
                command=command,
                timing=timing,
            )

        inserted = face_registry.register(userid, embeddings, qualities)

    logger.info(
        "/v1/vision/face/register userid=%s inserted=%d skipped=%d total_ms=%d",
        userid, inserted, skipped, timing.process_ms,
    )
    return build_response(
        None,
        module_id=MODULE_ID,
        module_name=MODULE_NAME,
        command=command,
        timing=timing,
        message=f"Registered {inserted} face(s) for {userid}",
        extra={"userid": userid, "registered": inserted, "skipped": skipped},
    )


@router.post("/list")
async def face_list() -> dict[str, Any]:
    command = "list"
    with Timing() as timing:
        faces = face_registry.list_userids()
    return build_response(
        None,
        module_id=MODULE_ID,
        module_name=MODULE_NAME,
        command=command,
        timing=timing,
        extra={"faces": faces},
    )


@router.post("/delete")
async def face_delete(
    userid: Annotated[str, Form(description="userid to remove from the registry")],
) -> dict[str, Any]:
    command = "delete"
    with Timing() as timing:
        existed = face_registry.delete(userid)
    msg = f"Deleted {userid}" if existed else f"{userid} not found"
    return build_response(
        None,
        module_id=MODULE_ID,
        module_name=MODULE_NAME,
        command=command,
        timing=timing,
        message=msg,
        extra={"userid": userid, "deleted": existed},
    )


@router.post("/match")
async def face_match(
    image1: Annotated[UploadFile, File(description="First face")],
    image2: Annotated[UploadFile, File(description="Second face")],
) -> dict[str, Any]:
    command = "match"
    with Timing() as timing:
        try:
            img1 = await read_image_upload(image1, field_name="image1")
            img2 = await read_image_upload(image2, field_name="image2")
        except Exception as exc:  # noqa: BLE001
            return error_response(str(exc), module_id=MODULE_ID, command=command, timing=timing)

        with timing.inference():
            f1 = face_pipeline.analyse(img1, min_confidence=0.0)
            f2 = face_pipeline.analyse(img2, min_confidence=0.0)

        if not f1 or not f2:
            return error_response(
                "Could not detect a face in both images",
                module_id=MODULE_ID,
                command=command,
                timing=timing,
            )

        # Pick best face per image — match() semantics in CodeProject compare
        # the single most prominent face in each input.
        b1 = max(f1, key=lambda f: f.det_score)
        b2 = max(f2, key=lambda f: f.det_score)
        e1 = b1.embedding
        e2 = b2.embedding
        n1 = e1 / max(float((e1 * e1).sum()) ** 0.5, 1e-12)
        n2 = e2 / max(float((e2 * e2).sum()) ** 0.5, 1e-12)
        similarity = float((n1 * n2).sum())

    return build_response(
        None,
        module_id=MODULE_ID,
        module_name=MODULE_NAME,
        command=command,
        timing=timing,
        extra={"similarity": round(similarity, 6)},
    )
