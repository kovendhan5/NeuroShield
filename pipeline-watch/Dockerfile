FROM python:3.11-slim

WORKDIR /app

# Install kubectl
RUN apt-get update && apt-get install -y curl gnupg && \
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && \
    chmod +x kubectl && mv kubectl /usr/local/bin/ && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --default-timeout=2000 --retries 10 --index-url https://pypi.org/simple/ -r requirements.txt

# App
COPY app.py .

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/health', timeout=3)"

EXPOSE 5000
CMD ["python", "app.py"]
