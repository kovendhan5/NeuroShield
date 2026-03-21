FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y curl build-essential && apt-get clean && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY backend/requirements.txt .
RUN pip install --upgrade pip setuptools wheel && pip install --default-timeout=2000 --retries 10 --index-url https://pypi.org/simple/ -r requirements.txt

# Copy application
COPY backend/app.py .
COPY frontend/public ./templates/public

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8888/api/health || exit 1

EXPOSE 8888

CMD ["python", "app.py"]
