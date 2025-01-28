#!/bin/bash

PORT=${API_PORT:-8000}

exec gunicorn -k uvicorn.workers.UvicornWorker -w 10 -b 0.0.0.0:${PORT} main:app