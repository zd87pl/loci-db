FROM python:3.11-slim

WORKDIR /app

# Build from the working tree, NOT PyPI: installing loci-stdb from PyPI here
# would silently pin the container to the last published release while running
# HEAD's server.py against it (a mismatch that broke /insert in the past).
# README.md must be copied because pyproject declares it as the package readme.
COPY pyproject.toml README.md ./
COPY loci/ ./loci/
RUN pip install --no-cache-dir . "fastapi[standard]>=0.100.0" "uvicorn[standard]>=0.24.0"

COPY server.py /app/server.py

ENV QDRANT_URL=http://qdrant:6333
ENV LOCI_VECTOR_SIZE=512
ENV LOCI_EPOCH_SIZE_MS=5000

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
