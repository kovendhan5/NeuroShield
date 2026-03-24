FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (including curl for healthcheck)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create neuroshield user (non-root execution)
RUN groupadd -r neuroshield && useradd -r -g neuroshield -u 1000 neuroshield

# Copy requirements and install
COPY --chown=neuroshield:neuroshield requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=neuroshield:neuroshield . .

# Create data/logs directories with proper permissions
RUN mkdir -p data logs && chown -R neuroshield:neuroshield /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/tmp/torch
ENV TORCHINDUCTOR_CACHE_DIR=/tmp/torch_cache

# Switch to non-root user
USER neuroshield

# Expose API port
EXPOSE 8000

# Health check (as neuroshield user)
HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run orchestrator with API
CMD ["python", "-m", "src.orchestrator.main"]
