#!/bin/bash

if [ -f .env ]; then
  # Filter out comments and empty lines from .env before exporting
  export $(grep -v '^#' .env | grep -v '^$' | xargs)
fi
uvicorn ml_eval.main:app --host 0.0.0.0 --port 8000