# MapOMOP API Server Dockerfile
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

# Install PyTorch with CUDA support and other dependencies
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
COPY api_server.py .
COPY src/ ./src/
COPY indexing/ ./indexing/

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    mkdir -p /app/logs && \
    chown -R appuser:appuser /app
USER appuser

# Environment variables
ENV ES_HOST=elasticsearch \
    ES_PORT=9200 \
    ES_USERNAME=elastic \
    ES_PASSWORD=snomed \
    SAPBERT_URL=http://sapbert:8000 \
    LOCAL_LLM_MODEL=qwen \
    LOCAL_LLM_PATH=/app/llm-models \
    MAPOMOP_HOST=0.0.0.0 \
    MAPOMOP_PORT=8080 \
    SCORING_MODE=local_llm \
    CONFIDENCE_THRESHOLD=0.5 \
    PYTHONPATH=/app:/app/src

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8080"]

