FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[speech,embeddings]"

COPY data ./data
# Shipped inside customer bundles by POST /v1/admin/tenants/{t}/recorder-bundle
COPY recorder ./recorder

# Run as non-root; STATE_DIR (deployment record, issued tokens) must be writable
# Mount points must exist and belong to app, or Docker creates the volumes
# root-owned and the model cache / state dir are not writable.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/state /home/app/.cache/huggingface \
    && chown -R app:app /app/state /home/app/.cache
USER app
VOLUME ["/app/state"]

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://localhost:8000/healthz')" || exit 1
CMD ["uvicorn", "callcenter.main:app", "--host", "0.0.0.0", "--port", "8000"]
