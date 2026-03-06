FROM python:3.12-slim

WORKDIR /app

RUN python -m pip install --no-cache-dir flask psutil

COPY app.py /app/app.py

EXPOSE 5000

CMD ["python", "/app/app.py"]
