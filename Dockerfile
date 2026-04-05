FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir "loci-stdb>=0.3.0" "fastapi[standard]>=0.100.0" "uvicorn[standard]>=0.24.0"

COPY server.py /app/server.py

ENV QDRANT_URL=http://qdrant:6333
ENV LOCI_VECTOR_SIZE=512
ENV LOCI_EPOCH_SIZE_MS=5000

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
