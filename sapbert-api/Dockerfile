# SapBERT API Server Dockerfile
# Multi-stage build for optimized image size

# ============================================================================
# Stage 1: Base image with CUDA support
# ============================================================================
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Seoul

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python

# ============================================================================
# Stage 2: Builder
# ============================================================================
FROM base AS builder

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt

# ============================================================================
# Stage 3: Production image
# ============================================================================
FROM base AS production

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY app.py .

# Create directories for models
RUN mkdir -p /app/models /app/hf_cache

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Environment variables
# SAPBERT_MODEL_PATH: 로컬 모델 경로 (볼륨 마운트 시 사용)
ENV SAPBERT_MODEL_PATH="/app/models" \
    SAPBERT_MODEL_NAME="cambridgeltl/SapBERT-from-PubMedBERT-fulltext" \
    SAPBERT_MAX_LENGTH=25 \
    SAPBERT_BATCH_SIZE=128 \
    SAPBERT_DEVICE=auto \
    SAPBERT_HOST=0.0.0.0 \
    SAPBERT_PORT=8000 \
    HF_HOME=/app/hf_cache \
    TRANSFORMERS_CACHE=/app/hf_cache

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

