#!/bin/bash
set -e

echo "Starting in environment: DISABLE_INTERNAL_OLLAMA=$DISABLE_INTERNAL_OLLAMA, OLLAMA_HOST=$OLLAMA_HOST"

# Check if OLLAMA_HOST contains 'chat:11434', indicating we're in docker-compose
if [[ "$OLLAMA_HOST" == *"chat:11434"* ]]; then
  echo "Detected docker-compose environment with external Ollama at $OLLAMA_HOST"
  # Force disable internal Ollama to avoid conflicts
  export DISABLE_INTERNAL_OLLAMA="true"
fi

# Check if we should disable internal Ollama
if [[ "$DISABLE_INTERNAL_OLLAMA" == "true" ]]; then
  echo "Internal Ollama service disabled. Using external Ollama service at $OLLAMA_HOST"
  # Check external Ollama is reachable
  echo "Checking connection to external Ollama API..."
  EXTERNAL_API_URL="${OLLAMA_HOST}/api/version"
  
  # Wait for external Ollama with timeout
  for i in {1..30}; do
    if curl -s "$EXTERNAL_API_URL" > /dev/null; then
      echo "External Ollama API is reachable at $OLLAMA_HOST"
      break
    fi
    
    if [ $i -eq 30 ]; then
      echo "Warning: External Ollama API did not respond in time, but continuing..."
    else
      echo "Waiting for external Ollama API... ($i seconds)"
      sleep 1
    fi
  done
else
  # Start Ollama service only if not disabled
  echo "Starting Ollama service..."
  # Use explicit bind address
  BIND_ADDR="localhost"
  echo "Starting local Ollama service on $BIND_ADDR"
  ollama serve > /dev/null 2>&1 &
  OLLAMA_PID=$!
  echo "Ollama service started with PID $OLLAMA_PID"

  # Wait for Ollama to start
  echo "Waiting for Ollama API to become available..."
  
  # Set direct API URL for service
  OLLAMA_API="${OLLAMA_HOST}/api/version"
  
  # Check if Ollama is ready with a timeout
  for i in {1..60}; do
    if curl -s "$OLLAMA_API" > /dev/null; then
      echo "Ollama API is ready after $i seconds"
      break
    fi
    
    if [ $i -eq 60 ]; then
      echo "Ollama API did not become available within 60 seconds"
      # Don't exit - FastAPI will handle unavailability and retry
    fi
    
    echo "Waiting for Ollama API... ($i seconds)"
    sleep 1
  done
  
  # Pull the model with a timeout if specified
  if [ -n "$OLLAMA_MODEL" ]; then
    echo "Pulling model $OLLAMA_MODEL..."
    timeout 120s ollama pull $OLLAMA_MODEL || echo "Model pull timed out or failed, continuing anyway"
    
    # Warm up the model with a simple query
    echo "Warming up model with a simple query..."
    echo "hello" | timeout 30s ollama run $OLLAMA_MODEL > /dev/null || echo "Model warm-up failed, continuing anyway"
  fi
fi

# Test Piper installation
echo "Testing Piper voice installation..."
if ! python3 -c "from piper import PiperVoice" 2>/dev/null; then
  echo "Warning: Piper installation issue detected, but continuing"
fi

# Start the FastAPI application
echo "Starting FastAPI application..."
exec uvicorn app:app --host 0.0.0.0 --port 8000 