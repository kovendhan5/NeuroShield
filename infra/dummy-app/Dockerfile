FROM python:3.12-slim

WORKDIR /app

RUN pip install --default-timeout=1000 --retries 5 flask psutil

COPY app.py /app/app.py

EXPOSE 5000

CMD ["python", "/app/app.py"]
