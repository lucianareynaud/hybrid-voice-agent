FROM python:3.11-slim
RUN apt-get update && apt-get install -y ffmpeg curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
# Install deps and download default Piper voice
RUN pip install --no-cache-dir -r requirements.txt \
 && piper --download-voice pt-br-joaquim-low
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
