#!/bin/bash
set -e

# Get the absolute path of the directory containing the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Change to the script directory
cd "$SCRIPT_DIR"

echo "===== Deploying PureVoice AI to voiceagent.lucianaferreira.pro ====="

# Create a prompt template for phi3:mini
# You can customize this to improve assistant performance
OLLAMA_SYSTEM_PROMPT=$(cat <<'EOF'
You are a helpful, harmless, and honest assistant.
Always provide accurate information and never make things up.
If you're unsure about something, you should acknowledge this 
and help the user find accurate information.
EOF
)

# Set environment variables for deployment
export OLLAMA_MODEL="phi3:mini"
export OLLAMA_SYSTEM_PROMPT

# Pull the latest code changes
echo "Pulling latest code changes..."
git pull

# Build and deploy with docker compose
echo "Building and deploying with Docker Compose..."
docker compose down
docker compose build
docker compose up -d

echo "Waiting for services to start..."
sleep 15

echo "===================================="
echo "PureVoice AI deployed successfully!"
echo "Visit: https://voiceagent.lucianaferreira.pro"
echo "====================================" 