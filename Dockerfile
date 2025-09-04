FROM python:3.12-slim
COPY requirements.txt requirements.txt
COPY setup.py setup.py
RUN pip install .
COPY backend/ backend
COPY emb_cache_20k/ emb_cache_20k
COPY embeddings/ embeddings
# WORKDIR "backend"
CMD uvicorn backend.app:app --port $PORT --host 0.0.0.0
