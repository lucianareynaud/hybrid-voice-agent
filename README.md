# PureVoice AI — Your voice, your AI. 100% private.

Welcome to **PureVoice AI**, the next-generation offline voice assistant.
100% private, fully customizable, and enterprise-grade.
This white-label solution is fully customizable for any industry or domain.
Run all components locally with no API keys, no service credits, and zero cloud dependencies.
Your audio and data never leave your machine—enterprise-grade privacy at your fingertips.

## System Requirements
 - Memory: Whisper 'tiny' uses ~250 MB RAM; Granite3.1-MoE 1B model uses ~2 GB RAM; allow ~3 GB total plus OS overhead (~4 GB+ recommended).
- CPU: 4+ cores. CPU-only inference can be several times real-time; for low-latency (<300 ms) consider GPU or remote inference.
- For low-memory setups (<8 GB), use remote chat/TTS backends.
- Docker and docker-compose installed; ports 8000 and 11434 accessible.

## Quick Start

### Docker Setup

```bash
# Step 1: Build the image (only needed once)
docker build -t voice-agent .

# Step 2: Run the container
docker run --rm -p 127.0.0.1:8000:8000 voice-agent
```
Access at: http://localhost:8000

> **Important**: Always access the application through `localhost` instead of IP addresses for microphone access to work properly. Browsers restrict microphone access on non-secure origins, but make exceptions for localhost.

## Available Models

PureVoice AI supports multiple LLM options:

| Model | Command | RAM Required | Description |
|-------|---------|--------------|-------------|
| Granite3.1-MoE 1B | `./start-app.sh` | ~2 GB | Efficient, balanced model with good reasoning abilities |
| Custom | Edit docker-compose.simple.yml | Varies | Set any Ollama-compatible model in the configuration |

> **Note**: The first run downloads the selected model, which may take time depending on your connection speed. Models are cached for subsequent runs.

## Minimum Specs
| Component | Model                | RAM Used |
|-----------|----------------------|----------|
| ASR       | whisper **tiny**     | ~250 MB  |
| LLM       | **Granite3.1-MoE 1B** | ~2 GB    |
| TTS       | piper en-us-ryan-low | ~150 MB  |
> Fits comfortably in 4 GB free RAM.

## Troubleshooting

### Microphone Access Issues
If you experience microphone access problems:

1. **Always use localhost**: Access the application through http://localhost:8000 rather than using IP addresses
2. **Check browser permissions**: Ensure your browser has permission to access your microphone
3. **Try a different browser**: Chrome and Edge tend to work best for local development

### Docker Networking Issues
If you encounter errors about address already in use:
1. Stop all running Docker containers: `docker-compose down --remove-orphans`
2. Check for processes using port 11434: `lsof -i :11434`
3. Edit docker-compose.simple.yml to use a different port if needed

## Cloudflare Tunnel
```bash
cloudflared tunnel run voice-agent
```
Exposes your local app to the internet via a secure Cloudflare Tunnel.

    