# ─────────────────────────────────────────────────────────────────────────────
# PlantDx — Dockerfile
#
# Multi-stage build:
#   Stage 1 (builder) : install dependencies into a virtualenv
#   Stage 2 (runtime) : copy only the venv + source, no build tools
#
# Usage
# ─────
# Build:
#   docker build -t plantdx:latest .
#
# Predict (single image):
#   docker run --rm \
#     -v /path/to/checkpoints:/app/checkpoints:ro \
#     -v /path/to/images:/app/images:ro \
#     plantdx:latest predict \
#       --checkpoint /app/checkpoints/best_model.pth \
#       --image      /app/images/leaf.jpg \
#       --top_k      3
#
# GPU support (requires nvidia-docker):
#   docker run --rm --gpus all \
#     -v /path/to/checkpoints:/app/checkpoints:ro \
#     -v /path/to/images:/app/images:ro \
#     plantdx:latest predict \
#       --checkpoint /app/checkpoints/best_model.pth \
#       --image_dir  /app/images/
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install CPU-only PyTorch first (keeps the image lean; swap index URL for GPU)
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Copy and install the package
COPY requirements.txt pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e "."


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title       = "PlantDx"
LABEL org.opencontainers.image.description = "EfficientNet-B4 plant disease classifier"
LABEL org.opencontainers.image.source      = "https://github.com/Pelex04/Diagnosis-plant-model"
LABEL org.opencontainers.image.licenses    = "MIT"

# Runtime system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy source and scripts
COPY src/     ./src/
COPY scripts/ ./scripts/

# Non-root user for security
RUN useradd -m -u 1000 plantdx && chown -R plantdx:plantdx /app
USER plantdx

# Healthcheck — verify the package is importable
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "from plantdx import PlantDiseaseClassifier; print('ok')" || exit 1

# Default: show predict CLI help
ENTRYPOINT ["python", "scripts/predict.py"]
CMD ["--help"]
