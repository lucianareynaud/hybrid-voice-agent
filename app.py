import os
import uuid
import base64
import tempfile
import subprocess
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
import httpx

# Optional ElevenLabs SDK
try:
    from elevenlabs import generate, set_api_key
    ELEVENLABS_AVAILABLE = True
    set_api_key(os.getenv("ELEVENLABS_API_KEY", ""))
except ImportError:
    ELEVENLABS_AVAILABLE = False

# Optional OpenAI SDK
try:
    import openai
    OPENAI_AVAILABLE = True
    openai.api_key = os.getenv("OPENAI_API_KEY", "")
except ImportError:
    OPENAI_AVAILABLE = False

# Load environment variables
load_dotenv()

# FastAPI init
app = FastAPI()

# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

# Load Whisper model once
model = WhisperModel(os.getenv("WHISPER_MODEL", "small"))

async def transcribe_audio(path: str) -> str:
    segments, _ = model.transcribe(path)
    return "".join(s.text for s in segments).strip()

async def chat_with_ollama(text: str) -> str:
    url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
    payload = {
        "model": os.getenv("OLLAMA_MODEL", "llama3:8b-instruct"),
        "messages": [{"role": "user", "content": text}],
        "stream": False
    }
    async with httpx.AsyncClient() as client:
        res = await client.post(url, json=payload)
        res.raise_for_status()
        data = res.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError):
            return str(data)

def local_tts(text: str) -> bytes:
    voice = os.getenv("PIPER_VOICE", "pt-br-joaquim-low")
    p = subprocess.run(
        ["piper", "--voice", voice],
        input=text.encode("utf-8"),
        capture_output=True
    )
    wav = p.stdout
    ff = subprocess.run(
        ["ffmpeg", "-i", "pipe:0", "-f", "mp3", "pipe:1", "-y", "-loglevel", "error"],
        input=wav,
        capture_output=True
    )
    return ff.stdout

def elevenlabs_tts(text: str) -> bytes:
    if not ELEVENLABS_AVAILABLE:
        raise HTTPException(500, "ElevenLabs SDK not installed or key missing")
    resp = generate(text=text, voice="Rachel", model="eleven_monolingual_v1", stream=False)
    return resp.read()

def openai_tts(text: str) -> bytes:
    if not OPENAI_AVAILABLE:
        raise HTTPException(500, "OpenAI SDK not installed or key missing")
    resp = openai.Audio.create(
        model="tts-1", voice="alloy", input=text, response_format="b64_json"
    )
    b64 = resp.get("audio")
    if not b64:
        raise HTTPException(500, "Bad response from OpenAI TTS")
    return base64.b64decode(b64)

@app.post("/process")
async def process(audio: UploadFile = File(...)):
    suffix = os.path.splitext(audio.filename)[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        transcript = await transcribe_audio(tmp_path)
        response_text = await chat_with_ollama(transcript)
    except Exception as e:
        raise HTTPException(500, str(e))

    backend = os.getenv("TTS_BACKEND", "local").lower()
    if backend == "local":
        audio_bytes = local_tts(response_text)
    elif backend == "elevenlabs":
        audio_bytes = elevenlabs_tts(response_text)
    elif backend == "openai":
        audio_bytes = openai_tts(response_text)
    else:
        raise HTTPException(400, f"Unknown TTS_BACKEND: {backend}")

    b64 = base64.b64encode(audio_bytes).decode("utf-8")
    return JSONResponse({
        "transcript": transcript,
        "response_text": response_text,
        "audio_format": "mp3",
        "audio_base64": b64
    })
