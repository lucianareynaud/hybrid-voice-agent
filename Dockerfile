# Base image
FROM python:3.11-slim

# Minimal runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg curl git \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama CLI (daemon will be started at runtime)
RUN curl -fsSL https://ollama.ai/install.sh | bash

# Create isolated virtual environment inside container
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Create voice directory
RUN mkdir -p /app/voices

# Copy application code
WORKDIR /app
COPY . .

# Install Python dependencies inside venv
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV WHISPER_MODEL=tiny
ENV OLLAMA_MODEL=phi4-mini
ENV TTS_BACKEND=local
ENV PIPER_VOICE=en-us-ryan-low

# Download the Piper voice model
RUN mkdir -p /root/.local/share/piper/voices && \
    mkdir -p /app/voices && \
    cd /root/.local/share/piper/voices && \
    curl -L -o en-us-ryan-low.onnx https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/low/en_US-ryan-low.onnx && \
    curl -L -o en-us-ryan-low.onnx.json https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/low/en_US-ryan-low.onnx.json && \
    cp en-us-ryan-low.onnx /app/voices/ && \
    cp en-us-ryan-low.onnx.json /app/voices/

EXPOSE 8000

# Start Ollama, wait for API, pull model once, then start FastAPI
CMD ["bash","-c","ollama serve & until curl -s http://localhost:11434 > /dev/null; do sleep 1; done; ollama pull ${OLLAMA_MODEL:-phi4-mini} || true; piper --help || echo 'Piper voice ready'; uvicorn app:app --host 0.0.0.0 --port 8000"]
