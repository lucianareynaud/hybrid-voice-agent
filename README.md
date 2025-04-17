# 🗣️  hybrid‑voice‑agent – Zero‑cost, browser‑based voice bot POC

## One‑click demo
1. `brew install ollama && ollama run llama3:8b-instruct`
2. `brew install piper && piper --download-voice pt-br-joaquim-low`
3. `pip install -r requirements.txt`
4. `uvicorn app:app --reload`
5. Open `http://localhost:8000` → hold the button and talk!

## Cloudflare Tunnel
```bash
cloudflared tunnel run voice-agent
```
_exposes_ https://voice.lucianaferreira.pro

## Switching to premium voices
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
