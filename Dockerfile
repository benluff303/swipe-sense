FROM python:3.12-slim
COPY requirements.txt requirements.txt
COPY setup.py setup.py
RUN pip install .
COPY backend/ backend
# WORKDIR "backend"
CMD uvicorn backend.app:app --port $PORT --host 0.0.0.0
