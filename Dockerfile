FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ffmpeg curl procps bash && \
    rm -rf /var/lib/apt/lists/*

# Install Ollama CLI
RUN curl -fsSL https://ollama.ai/install.sh | bash

# Start Ollama daemon, wait 3s, pull model, then clean up
RUN ollama serve & sleep 3 && ollama pull ${OLLAMA_MODEL:-phi3:mini} && pkill ollama

WORKDIR /app
# Copy application code
COPY . .

# Install Python dependencies and download Piper voice
RUN pip install --no-cache-dir -r requirements.txt && \
    piper --download-voice ${PIPER_VOICE:-pt-br-joaquim-low}

# Expose application and Ollama ports
EXPOSE 8000 11434

# Entrypoint: launch Ollama and FastAPI together
CMD bash -c "ollama serve & uvicorn app:app --host 0.0.0.0 --port 8000"
