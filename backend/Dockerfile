FROM python:3.11-slim

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN grep -E '^(torch|torchvision)[<=>]' requirements.txt > /tmp/torch-requirements.txt \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu -r /tmp/torch-requirements.txt \
    && pip install --no-cache-dir -r requirements.txt
