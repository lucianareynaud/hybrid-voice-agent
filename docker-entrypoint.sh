#!/bin/bash
set -e

echo "Starting in environment: DISABLE_INTERNAL_OLLAMA=$DISABLE_INTERNAL_OLLAMA, OLLAMA_HOST=$OLLAMA_HOST"

# Check if OLLAMA_HOST contains 'chat:11434', indicating we're in docker-compose
if [[ "$OLLAMA_HOST" == *"chat:11434"* ]]; then
  echo "Detected docker-compose environment with external Ollama at $OLLAMA_HOST"
  # Force disable internal Ollama to avoid conflicts
  export DISABLE_INTERNAL_OLLAMA="true"
fi

# Create Granite modelfile for the purevoice-granite model
echo "Creating Granite modelfile..."
cat > /tmp/granite-modelfile.txt << 'EOL'
FROM granite3.1-moe:1b

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
  
  # Pull and create the custom Granite model if configured
  if [ "$OLLAMA_MODEL" == "purevoice-granite" ]; then
    echo "Pulling base model granite3.1-moe:1b..."
    timeout 300s ollama pull granite3.1-moe:1b || echo "Base model pull timed out or failed, continuing anyway"

    echo "Creating custom model purevoice-granite..."
    ollama create purevoice-granite -f /tmp/granite-modelfile.txt

    # Warm up the custom model with a simple query
    echo "Warming up custom model with a simple query..."
    echo "hello" | timeout 30s ollama run purevoice-granite > /dev/null || echo "Model warm-up failed, continuing anyway"
  else
    # Pull the model with a timeout if specified
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