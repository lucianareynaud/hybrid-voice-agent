# PureVoice AI — Your voice, your AI. 100% private.

Welcome to **PureVoice AI**, the next-generation offline voice assistant.
100% private, fully customizable, and enterprise-grade.
This white-label solution is fully customizable for any industry or domain.
Run all components locally with no API keys, no service credits, and zero cloud dependencies.
Your audio and data never leave your machine—enterprise-grade privacy at your fingertips.

## System Requirements
- Memory: Whisper 'small' uses ~2–3 GB RAM; Ollama llama3:8b-instruct uses ~10 GB; allow ~14 GB total plus OS overhead (~16 GB+ recommended).
- CPU: 4+ cores. CPU-only inference can be several times real-time; for low-latency (<300 ms) consider GPU or remote inference.
- For low-memory setups (<8 GB), use `WHISPER_MODEL=tiny` (~200 MB) and/or remote chat/TTS backends.
- Docker installed; ports 8000 (agent) and 11434 (chat) accessible.

## Quick Start

### Option 1: Single Docker Image (Recommended)
Build and run everything in one container — no additional install steps.
```bash
docker build -t voice-agent .
docker run --rm -p 8000:8000 voice-agent        # nothing else to install
```
> Heads-up: On the first run the container downloads the phi3:mini model (~3.5 GB). Subsequent starts are instant because the model is cached inside the container layer. No API keys required.
Open http://localhost:8000 in your browser and hold the button to talk.

### Option 2: Manual Local Install

#### Linux / macOS
1. Install Ollama chat model:
   ```bash
   # macOS
   brew install ollama
   # or Linux (Homebrew or see https://ollama.ai/docs/installation)
   ```
2. Start the chat service:
   ```bash
   ollama run ${OLLAMA_MODEL:-phi3:mini}
   ```
3. Install Piper TTS engine:
   ```bash
   # macOS
   brew install piper
   # or Linux (cargo install or binaries from https://github.com/rhasspy/piper)
   ```
4. Download a local voice:
   ```bash
   piper --download-voice ${PIPER_VOICE:-pt-br-joaquim-low}
   ```
5. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
6. Launch the FastAPI app:
   ```bash
   uvicorn app:app --reload
   ```
7. Open http://localhost:8000 and hold the button to talk.

#### Windows
Use Windows Subsystem for Linux (WSL) and follow the Linux instructions above,
or run the single Docker image (Option 1) for a fully automated setup.

## Minimum Specs
| Component | Model                 | RAM Used |
|-----------|-----------------------|----------|
| ASR       | whisper **tiny**      | ~0.5 GB   |
| LLM       | **phi3:mini** (3.8 B) | ~3.5 GB   |
| TTS       | piper pt‑br‑joaquim   | ~0.2 GB   |
> Fits comfortably in 8 GB free RAM.
> To upgrade quality, bump **WHISPER_MODEL=small** and **OLLAMA_MODEL=llama3:8b-instruct** (needs ≈14 GB free).

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
- WebSocket streaming (<300 ms latency)
- VAD to auto‑detect end‑of‑utterance
- Twilio Media Streams phone‑number bridge
- LoRA fine‑tuning for custom domain-specific language
- Front‑end subtitles with word‑level timing

MIT License
