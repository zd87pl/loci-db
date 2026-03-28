FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY demo/requirements.txt /app/demo/requirements.txt
RUN pip install --no-cache-dir -r demo/requirements.txt

# Copy LOCI source (use local package rather than PyPI)
COPY loci/ /app/loci/
COPY pyproject.toml /app/pyproject.toml
RUN pip install --no-cache-dir -e .

# Copy demo app
COPY demo/ /app/demo/

ENV PORT=8000
EXPOSE 8000

CMD ["uvicorn", "demo.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
