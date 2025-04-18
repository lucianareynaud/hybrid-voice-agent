#!/bin/bash
set -e

echo "===== Starting PureVoice AI with Qwen2.5 3B model ====="

# Stop any existing containers
echo "Stopping any existing containers..."
docker compose down

# Create custom Modelfile for Qwen
echo "Creating custom Modelfile for Qwen2.5..."
cat > qwen-modelfile.txt << 'EOL'
FROM qwen2.5:3b

SYSTEM """
You are a helpful voice assistant called PureVoice. 
You have access to a vast knowledge base and can answer questions on many topics.
Never say you don't have certain capabilities or access to information.
Always try to provide a helpful, concise response.
If you truly don't know something, you can say "I'm not sure about that" instead of saying you lack capabilities.
"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "<|endoftext|>"
EOL

# Start services
echo "Starting services..."
OLLAMA_MODEL=purevoice-qwen docker compose up -d

# Wait for Ollama to be ready (shorter timeout)
echo "Waiting for Ollama service to initialize..."
for i in {1..15}; do
  if curl -s http://localhost:11434/api/version > /dev/null; then
    echo "✅ Ollama service is ready!"
    break
  fi
  echo "Waiting for Ollama API... ($i/15)"
  sleep 1
done

# Pull the base model first
echo "Downloading qwen2.5:3b base model..."
docker exec hybrid-voice-agent-cursor-ollama-1 ollama pull qwen2.5:3b

# Copy the Modelfile into the container and create our custom model
echo "Creating custom purevoice-qwen model..."
docker cp qwen-modelfile.txt hybrid-voice-agent-cursor-ollama-1:/tmp/qwen-modelfile.txt
docker exec hybrid-voice-agent-cursor-ollama-1 ollama create purevoice-qwen -f /tmp/qwen-modelfile.txt

# Warm up the model with a very short prompt
echo "Warming up the model..."
docker exec hybrid-voice-agent-cursor-ollama-1 ollama run purevoice-qwen "Hi" --verbose=false

# Check if the app is running (shorter timeout)
echo "Checking if the app is ready..."
for i in {1..10}; do
  if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Voice app is ready! Access at: http://localhost:8000"
    echo ""
    echo "Using model: purevoice-qwen (custom Qwen2.5 3B model)"
    echo "For production deployment at voiceagent.lucianaferreira.pro:"
    echo "1. Configure your reverse proxy (nginx/caddy) to point to this service"
    echo "2. Ensure ports 8000 and 11434 are not exposed to the internet directly"
    echo "3. Consider running 'docker update --cpus=\"4\" --memory=8G hybrid-voice-agent-cursor-ollama-1' to allocate more resources"
    exit 0
  fi
  echo "Waiting for app to be ready... ($i/10)"
  sleep 1
done

echo "Something went wrong. Showing logs:"
docker compose logs
echo "Try accessing http://localhost:8000 manually, or run 'docker compose logs' for detailed logs." 