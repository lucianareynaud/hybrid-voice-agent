#!/bin/bash
set -e

echo "===== Starting PureVoice AI with Mistral model ====="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
  echo "Error: Docker is not running or not accessible. Please start Docker and try again."
  exit 1
fi

# Clean up any existing containers
echo "Stopping any existing containers..."
docker-compose -f docker-compose.simple.yml down --remove-orphans

# Start fresh containers
echo "Starting Ollama and Voice App..."
docker-compose -f docker-compose.simple.yml up -d

# Display logs for Ollama to show progress
echo "Displaying Ollama logs (will timeout after 30 seconds)..."
timeout 30 docker-compose -f docker-compose.simple.yml logs -f ollama || true

# Check the health of the Ollama container
echo "Checking Ollama container health..."
if [[ $(docker inspect --format='{{.State.Health.Status}}' $(docker-compose -f docker-compose.simple.yml ps -q ollama 2>/dev/null) 2>/dev/null) != "healthy" ]]; then
  echo "Warning: Ollama container is not yet healthy. This may take a few minutes for initial model download."
  echo "You can check logs with: docker-compose -f docker-compose.simple.yml logs -f ollama"
fi

# Wait for services to be ready
echo "Waiting for Voice App to be ready..."
max_retries=20
for i in $(seq 1 $max_retries); do
  if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ App is ready! Available at: http://localhost:8000"
    echo "You can check logs with: docker-compose -f docker-compose.simple.yml logs -f"
    exit 0
  fi
  
  # Check if containers are running
  if ! docker-compose -f docker-compose.simple.yml ps | grep -q "voice-app.*Up"; then
    echo "❌ Error: Voice App container is not running."
    echo "Checking logs for errors..."
    docker-compose -f docker-compose.simple.yml logs voice-app
    echo ""
    echo "Try running: docker-compose -f docker-compose.simple.yml up -d voice-app --force-recreate"
    exit 1
  fi
  
  echo "Waiting for app to start... ($i/$max_retries)"
  sleep 3
done

echo "❌ App didn't start in expected time. Displaying logs:"
docker-compose -f docker-compose.simple.yml logs voice-app
echo ""
echo "For full logs, run: docker-compose -f docker-compose.simple.yml logs -f"
echo "You may need to wait longer for the model to download if this is your first time running." 