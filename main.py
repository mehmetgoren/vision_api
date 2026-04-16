import os

import uvicorn


def start_web_server():
    port = 8014
    if os.environ.get('WEBSERVER_ENVIRONMENT') == 'Container':
        print('WEBSERVER_ENVIRONMENT: Container')
        return uvicorn.run('api.methods:app', host='0.0.0.0', port=port, log_level='info', reload=True)
    return uvicorn.run('api.methods:app', host='0.0.0.0', port=port, log_level='info', reload=True)


def main():
    start_web_server()

# sudo apt install tesseract-ocr-all
if __name__ == '__main__':
    main()
