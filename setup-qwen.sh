#!/bin/bash
set -e

echo "===== Setting up Qwen2.5 3B model ====="

# Create Qwen modelfile
echo "Creating Qwen modelfile..."
cat > /tmp/qwen-modelfile.txt << 'EOL'
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

# Pull the base model
echo "Downloading qwen2.5:3b base model..."
ollama pull qwen2.5:3b

# Create the custom model
echo "Creating purevoice-qwen model..."
ollama create purevoice-qwen -f /tmp/qwen-modelfile.txt

# Warm up the model
echo "Warming up the model..."
ollama run purevoice-qwen "Write a short greeting" --verbose=false

echo "===== Qwen2.5 3B model setup complete =====" 