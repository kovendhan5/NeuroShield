FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create neuroshield user (non-root execution)
RUN groupadd -r neuroshield && useradd -r -g neuroshield -u 1000 neuroshield

# Copy requirements and install
COPY --chown=neuroshield:neuroshield requirements.txt .

# Pre-install PyTorch CPU-only to optimize image size
RUN pip install --no-cache-dir "torch>=2.1.0,<3" --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=neuroshield:neuroshield . .

# Create data/logs directories with proper permissions
RUN mkdir -p data logs /tmp/torch /tmp/torch_cache /tmp/huggingface && \
    chown -R neuroshield:neuroshield /app /tmp/torch /tmp/torch_cache /tmp/huggingface

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/tmp/torch
ENV TORCHINDUCTOR_CACHE_DIR=/tmp/torch_cache
ENV HF_HOME=/tmp/huggingface

# Switch to non-root user
USER neuroshield

# Expose API port
EXPOSE 8000

# Health check (as neuroshield user)
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD test -f /tmp/orchestrator_alive || exit 1

# Run orchestrator with API
CMD ["python", "-m", "src.orchestrator.main"]
