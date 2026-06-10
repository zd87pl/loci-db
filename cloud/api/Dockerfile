FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (cache layer)
COPY cloud/api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install loci package from repo root
COPY loci/ ./loci/
COPY pyproject.toml ./pyproject.toml
COPY README.md ./README.md
RUN pip install --no-cache-dir .

# Copy API source
COPY cloud/api/auth.py ./auth.py
COPY cloud/api/server.py ./server.py

# Non-root user
RUN useradd --no-create-home --shell /bin/false appuser
USER appuser

EXPOSE 8000

# --proxy-headers + --forwarded-allow-ips: the machine is only reachable via
# Fly.io's proxy, so trust its X-Forwarded-For and surface the real client IP to
# request.client.host. Without this every client appears as the proxy IP and the
# per-IP auth-failure throttle collapses into a single shared bucket.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips", "*"]
