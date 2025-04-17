# hybrid-voice-agent – Zero-cost, browser-based voice bot POC

Welcome to **hybrid-voice-agent**, the world’s first fully offline, privacy-first browser-based voice bot.
Run all components locally with no API keys, no service credits, and zero cloud dependencies.
Your audio and data never leave your machine—enterprise-grade privacy at your fingertips.

## System Requirements
- Memory: Whisper 'small' uses ~2–3 GB RAM; Ollama llama3:8b-instruct uses ~10 GB; allow ~14 GB total plus OS overhead (~16 GB+ recommended).
- CPU: 4+ cores. CPU-only inference can be several times real-time; for low-latency (<300 ms) consider GPU or remote inference.
- For low-memory setups (<8 GB), use `WHISPER_MODEL=tiny` (~200 MB) and/or remote chat/TTS backends.
- Docker & Docker Compose installed; ports 8000 (agent) and 11434 (chat) accessible.

## One‑click demo
1. `brew install ollama && ollama run llama3:8b-instruct`
2. `brew install piper && piper --download-voice pt-br-joaquim-low`
3. `pip install -r requirements.txt`
4. `uvicorn app:app --reload`
5. Open `http://localhost:8000` → hold the button and talk!

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
_exposes_ https://voice.lucianaferreira.pro

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
- LoRA fine‑tuning for Amil domain language
- Front‑end subtitles with word‑level timing

MIT License
