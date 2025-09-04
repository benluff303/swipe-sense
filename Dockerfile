FROM python:3.12-slim
COPY requirements.txt requirements.txt
COPY setup.py setup.py
RUN pip install .
COPY backend/ backend
COPY emb_cache_20k/ emb_cache_20k
COPY embeddings/ embeddings
COPY json_files/ json_files
COPY users/ users
# WORKDIR /backend
CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
