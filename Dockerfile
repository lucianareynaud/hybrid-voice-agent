# Base image
FROM python:3.11-slim

# Minimal runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama CLI (daemon will be started at runtime)
RUN curl -fsSL https://ollama.ai/install.sh | bash

# Create isolated virtual environment inside container
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy application code
WORKDIR /app
COPY . .

# Install Python dependencies inside venv
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

# Start Ollama, wait for API, pull model once, then start FastAPI
CMD ["bash","-c","ollama serve & until curl -s http://localhost:11434 > /dev/null; do sleep 1; done; ollama pull ${OLLAMA_MODEL:-phi3:mini} || true; uvicorn app:app --host 0.0.0.0 --port 8000"]
