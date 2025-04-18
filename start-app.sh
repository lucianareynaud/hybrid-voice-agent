#!/bin/bash
set -e

echo "===== Starting PureVoice AI with Mistral model ====="

# Stop any existing containers
echo "Stopping any existing containers..."
docker-compose -f docker-compose.simple.yml down

# Start services
echo "Starting services..."
docker-compose -f docker-compose.simple.yml up -d

# Wait for Ollama to be ready
echo "Waiting for Ollama service to initialize..."
for i in {1..30}; do
  if curl -s http://localhost:11434/api/version > /dev/null; then
    echo "✅ Ollama service is ready!"
    break
  fi
  echo "Waiting for Ollama API... ($i/30)"
  sleep 2
done

# Pull the model
echo "Downloading and initializing Mistral model (this may take a few minutes)..."
docker exec hybrid-voice-agent-cursor-ollama-1 ollama pull mistral

# Warm up the model
echo "Warming up the model..."
docker exec hybrid-voice-agent-cursor-ollama-1 ollama run mistral "Hello" --verbose=false

# Check if the app is running
echo "Checking if the app is ready..."
for i in {1..15}; do
  if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Voice app is ready! Access at: http://localhost:8000"
    exit 0
  fi
  echo "Waiting for app to be ready... ($i/15)"
  sleep 2
done

echo "Something went wrong. Showing logs:"
docker-compose -f docker-compose.simple.yml logs
echo "Try accessing http://localhost:8000 manually, or run 'docker-compose -f docker-compose.simple.yml logs' for detailed logs." 