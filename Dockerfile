# Base image
FROM python:3.11-slim

# Minimal runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg curl git netcat-openbsd iputils-ping net-tools \
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

# Set environment variables for faster startup
ENV WHISPER_MODEL=tiny
ENV OLLAMA_MODEL=mistral
ENV TTS_BACKEND=local
ENV PIPER_VOICE=en-us-ryan-low
ENV OLLAMA_HOST=http://localhost:11434
ENV MAX_RETRIES=5
ENV RETRY_DELAY=1.0
ENV DISABLE_INTERNAL_OLLAMA=false
ENV LOG_LEVEL=DEBUG

# Download the Piper voice model
RUN mkdir -p /root/.local/share/piper/voices && \
    mkdir -p /app/voices && \
    cd /root/.local/share/piper/voices && \
    curl -L -o en-us-ryan-low.onnx https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/low/en_US-ryan-low.onnx && \
    curl -L -o en-us-ryan-low.onnx.json https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/low/en_US-ryan-low.onnx.json && \
    cp en-us-ryan-low.onnx /app/voices/ && \
    cp en-us-ryan-low.onnx.json /app/voices/

# Add a health check to quickly indicate readiness
HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Start script with optimized startup
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Use the entrypoint script
CMD ["/app/docker-entrypoint.sh"]
