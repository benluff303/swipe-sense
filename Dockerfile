FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY backend/ backend
# WORKDIR "backend"
CMD uvicorn app:app --port $PORT --host 0.0.0.0
