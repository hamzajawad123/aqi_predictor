#!/bin/sh
set -e
uvicorn api.main:app --host 0.0.0.0 --port 8000 &
streamlit run app/Home.py --server.port=8501 --server.address=0.0.0.0 &
trap 'kill $(jobs -p) 2>/dev/null; wait' TERM INT
wait
