FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ffmpeg curl bash && \
    rm -rf /var/lib/apt/lists/*

# Install Ollama CLI and pull default low-memory model
RUN curl -fsSL https://ollama.ai/install.sh | bash && \
    ollama pull ${OLLAMA_MODEL:-phi3:mini}

WORKDIR /app
# Copy application code
COPY . .

# Install Python dependencies and download Piper voice
RUN pip install --no-cache-dir -r requirements.txt && \
    piper --download-voice ${PIPER_VOICE:-pt-br-joaquim-low}

# Expose application and Ollama ports
EXPOSE 8000 11434

# Entrypoint: run Ollama chat model in background and start FastAPI
ENTRYPOINT ["bash", "-lc", \
  "ollama run ${OLLAMA_MODEL:-phi3:mini} & \
   exec uvicorn app:app --host 0.0.0.0 --port 8000"]
