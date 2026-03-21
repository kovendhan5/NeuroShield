FROM python:3.12-slim

WORKDIR /app

RUN pip install --default-timeout=2000 --retries 10 --index-url https://pypi.org/simple/ flask psutil

COPY app.py /app/app.py

EXPOSE 5000

CMD ["python", "/app/app.py"]
