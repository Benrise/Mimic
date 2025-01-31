#!/bin/bash

PORT=${API_PORT:-8000}

exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --reload
# exec gunicorn -k uvicorn.workers.UvicornWorker -w 4 -b 0.0.0.0:${PORT} main:app