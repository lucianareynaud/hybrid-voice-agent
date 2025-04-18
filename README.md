# PureVoice AI — Your voice, your AI. 100% private.

Welcome to **PureVoice AI**, the next-generation offline voice assistant.
100% private, fully customizable, and enterprise-grade.
This white-label solution is fully customizable for any industry or domain.
Run all components locally with no API keys, no service credits, and zero cloud dependencies.
Your audio and data never leave your machine—enterprise-grade privacy at your fingertips.

## System Requirements
- Memory: Whisper 'tiny' uses ~250 MB RAM; Qwen 3B model uses ~4 GB RAM; allow ~6 GB total plus OS overhead (~8 GB+ recommended).
- CPU: 4+ cores. CPU-only inference can be several times real-time; for low-latency (<300 ms) consider GPU or remote inference.
- For low-memory setups (<8 GB), use remote chat/TTS backends.
- Docker and docker-compose installed; ports 8000 and 11434 accessible.

## Quick Start

### Using the Simple Docker Setup (Recommended)

```bash
# Run the setup script for Qwen model
./start-app.sh

# Or manually start with docker-compose
docker-compose -f docker-compose.simple.yml up -d
```

Access at: http://localhost:8000

### Alternative: Single Container Setup

```bash
# Step 1: Build the image (only needed once)
docker build -t voice-agent .

# Step 2: Run the container
docker run --rm -p 127.0.0.1:8000:8000 voice-agent
```

> **Important**: Always access the application through `localhost` instead of IP addresses for microphone access to work properly. Browsers restrict microphone access on non-secure origins, but make exceptions for localhost.

## Available Models

PureVoice AI supports multiple LLM options:

| Model | Command | RAM Required | Description |
|-------|---------|--------------|-------------|
| Qwen 3B | `./start-app.sh` | ~4 GB | Efficient, knowledge-rich model with good reasoning abilities |
| Custom | Edit docker-compose.simple.yml | Varies | Set any Ollama-compatible model in the configuration |

> **Note**: The first run downloads the selected model, which may take time depending on your connection speed. Models are cached for subsequent runs.

## Minimum Specs
| Component | Model                | RAM Used |
|-----------|----------------------|----------|
| ASR       | whisper **tiny**     | ~250 MB  |
| LLM       | **Qwen 3B**          | ~4 GB    |
| TTS       | piper en-us-ryan-low | ~150 MB  |
> Fits comfortably in 8 GB free RAM.

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

## Premium Voices (Optional)
By default, the demo runs 100% offline with no API keys or service credits required. To enable enhanced cloud-based voices, set one of the following:
```bash
export TTS_BACKEND=elevenlabs  # plus ELEVENLABS_API_KEY
# or
export TTS_BACKEND=openai      # plus OPENAI_API_KEY
```

## Roadmap (next steps)
- WebSocket streaming (<300 ms latency)
- VAD to auto‑detect end‑of‑utterance
- Twilio Media Streams phone‑number bridge
- LoRA fine‑tuning for custom domain-specific language
- Front‑end subtitles with word‑level timing

Apache 2.0 License
