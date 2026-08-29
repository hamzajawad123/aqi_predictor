# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY app/requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt \
    && pip install fastapi==0.115.0 "uvicorn[standard]==0.30.6"

COPY src/ ./src/
COPY api/ ./api/
COPY app/ ./app/
COPY .streamlit ./.streamlit
COPY docker-entrypoint.sh ./docker-entrypoint.sh
RUN chmod +x docker-entrypoint.sh

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health'); urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["./docker-entrypoint.sh"]
