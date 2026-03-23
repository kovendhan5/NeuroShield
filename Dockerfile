FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create data directory
RUN mkdir -p data logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/tmp/torch
ENV TORCHINDUCTOR_CACHE_DIR=/tmp/torch_cache

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run orchestrator with API
CMD ["python", "-m", "src.orchestrator.main"]
