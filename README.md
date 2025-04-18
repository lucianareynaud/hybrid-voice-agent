# PureVoice AI — Your voice, your AI. 100% private.

Welcome to **PureVoice AI**, the next-generation offline voice assistant.
100% private, fully customizable, and enterprise-grade.
This white-label solution is fully customizable for any industry or domain.
Run all components locally with no API keys, no service credits, and zero cloud dependencies.
Your audio and data never leave your machine—enterprise-grade privacy at your fingertips.

## System Requirements
- Memory: Whisper 'small' uses ~2–3 GB RAM; Ollama llama3:8b-instruct uses ~10 GB; allow ~14 GB total plus OS overhead (~16 GB+ recommended).
- CPU: 4+ cores. CPU-only inference can be several times real-time; for low-latency (<300 ms) consider GPU or remote inference.
- For low-memory setups (<8 GB), use `WHISPER_MODEL=tiny` (~200 MB) and/or remote chat/TTS backends.
- Docker installed; port 8000 accessible.

## Quick Start

### Just 2 Simple Steps

```bash
# Step 1: Build the image (only needed once)
docker build -t voice-agent .

# Step 2: Run the container
docker run --rm -p 127.0.0.1:8000:8000 voice-agent
```

Access at: http://localhost:8000

> **Important**: Always access the application through `localhost` instead of IP addresses for microphone access to work properly. Browsers restrict microphone access on non-secure origins, but make exceptions for localhost.

> **Note**: On the first run, the container downloads the phi4-mini model (~3.5 GB). Subsequent starts are faster because the model is cached inside the container layer.

## Minimum Specs
| Component | Model                 | RAM Used |
|-----------|-----------------------|----------|
| ASR       | whisper **tiny**      | ~0.5 GB   |
| LLM       | **phi3:mini** (3.8 B) | ~3.5 GB   |
| TTS       | piper en-us-ryan-low  | ~0.2 GB   |
> Fits comfortably in 8 GB free RAM.
> To upgrade quality, bump **WHISPER_MODEL=small** and **OLLAMA_MODEL=llama3:8b-instruct** (needs ≈14 GB free).

## Troubleshooting

### Microphone Access Issues
If you experience microphone access problems:

1. **Always use localhost**: Access the application through http://localhost:8000 rather than using IP addresses
2. **Check browser permissions**: Ensure your browser has permission to access your microphone
3. **Try a different browser**: Chrome and Edge tend to work best for local development

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

MIT License
