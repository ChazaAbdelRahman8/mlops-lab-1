# ============================================================
# Stage 1 - Builder
# ============================================================

FROM python:3.13-slim AS builder

WORKDIR /app

# More tolerant network settings
ENV UV_HTTP_TIMEOUT=600
ENV UV_HTTP_RETRIES=10

# Install uv
RUN pip install --no-cache-dir uv

# Create the virtual environment
RUN uv venv /opt/venv

# ------------------------------------------------------------
# Install CPU-only PyTorch for model serving
#
# The host machine can still use CUDA for training.
# The Docker container does not need CUDA for this lab.
# ------------------------------------------------------------

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
    --python /opt/venv/bin/python \
    --torch-backend=cpu \
    torch torchvision

# ------------------------------------------------------------
# Install only the packages needed by the serving API
# ------------------------------------------------------------

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
    --python /opt/venv/bin/python \
    mlflow \
    fastapi \
    uvicorn \
    python-multipart \
    pillow \
    numpy


# ============================================================
# Stage 2 - Runtime
# ============================================================

FROM python:3.13-slim AS runtime

WORKDIR /app

# Copy only the finished Python environment
COPY --from=builder /opt/venv /opt/venv

# Copy application source
COPY src/ ./src/

# Use packages from our virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Flush Python logs immediately
ENV PYTHONUNBUFFERED=1

# Local default.
# On Windows this will be overridden at docker run time with:
# http://host.docker.internal:5000
ENV MLFLOW_TRACKING_URI="http://127.0.0.1:5000"

EXPOSE 8000

CMD ["uvicorn", "src.food11.serve:app", "--host", "0.0.0.0", "--port", "8000"]