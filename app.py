import os
import uuid
import base64
import tempfile
import subprocess
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Low-memory model defaults
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny")
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL", "phi3:mini")
OLLAMA_URL    = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")

# FastAPI init
app = FastAPI(
    title="Voice Assistant API",
    description="A hybrid voice assistant using Whisper, Ollama, and TTS",
    version="1.0.0"
)

# Add CORS middleware
allowed_origins = [
    "http://localhost:8000",
    "http://0.0.0.0:8000",
    "https://localvoice.lucianaferreira.pro"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Load Whisper model once
try:
    model = WhisperModel(WHISPER_MODEL, device="auto")
    logger.info(f"Loaded Whisper model: {WHISPER_MODEL}")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {str(e)}")
    model = None

async def transcribe_audio(path: str) -> str:
    if not model:
        raise HTTPException(500, "Whisper model failed to load")
    try:
        segments, _ = model.transcribe(path)
        return "".join(s.text for s in segments).strip()
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(500, f"Transcription failed: {str(e)}")

async def chat_with_ollama(text: str) -> str:
    url = OLLAMA_URL
    # Low-memory LLM default
    messages = [{"role": "user", "content": text}]
    payload = {"model": OLLAMA_MODEL, "stream": False, "messages": messages}
    
    logger.info(f"Sending request to Ollama API: {url}")
    logger.info(f"Using model: {OLLAMA_MODEL}")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            res = await client.post(url, json=payload)
            res.raise_for_status()
            data = res.json()
            
            logger.info(f"Ollama API response structure: {list(data.keys())}")
            
            try:
                # Handle different Ollama response formats
                if "message" in data and "content" in data["message"]:
                    logger.info("Using message.content format")
                    return data["message"]["content"].strip()
                elif "response" in data:
                    logger.info("Using response format")
                    return data["response"].strip()
                elif "choices" in data and len(data["choices"]) > 0:
                    logger.info("Using choices[0].message.content format")
                    return data["choices"][0]["message"]["content"].strip()
                else:
                    # Fallback to trying to extract any text we can find
                    logger.error(f"Unknown Ollama response format: {data}")
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if isinstance(value, str) and len(value) > 0:
                                logger.info(f"Using fallback key: {key}")
                                return value.strip()
                    # If we can't find anything useful, return the stringified data
                    return str(data)
            except (KeyError, IndexError) as e:
                logger.error(f"Unexpected Ollama response format: {str(e)}")
                # Return raw data as a fallback
                return str(data)
    except httpx.HTTPError as e:
        logger.error(f"Ollama API error: {str(e)}")
        raise HTTPException(503, f"LLM service unavailable: {str(e)}")
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(500, f"Failed to generate response: {str(e)}")

def local_tts(text: str) -> bytes:
    voice = os.getenv("PIPER_VOICE", "en-us-ryan-low")
    voice_path = f"/root/.local/share/piper/voices/{voice}.onnx"
    voice_config = f"/root/.local/share/piper/voices/{voice}.onnx.json"
    
    # Debug voice file existence
    if not os.path.exists(voice_path):
        logger.error(f"Voice model file not found at {voice_path}")
        # Try alternative locations
        alt_path = f"/app/voices/{voice}.onnx"
        if os.path.exists(alt_path):
            logger.info(f"Found voice model at alternative path: {alt_path}")
            voice_path = alt_path
            voice_config = f"/app/voices/{voice}.onnx.json"
        else:
            # List available files to help debug
            try:
                os.makedirs("/root/.local/share/piper/voices", exist_ok=True)
                logger.info(f"Files in /root/.local/share/piper/voices: {os.listdir('/root/.local/share/piper/voices')}")
            except Exception as e:
                logger.error(f"Error listing voice files: {str(e)}")
            raise HTTPException(500, f"Voice model file not found: {voice_path}")
    
    try:
        logger.info(f"Using voice model: {voice_path}")
        
        # Create temporary files for piper output
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_temp:
            wav_path = wav_temp.name
        
        # Use temporary file for output instead of stdout
        p = subprocess.run(
            ["piper", "--model", voice_path, "--output-file", wav_path],
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=30
        )
        
        if p.returncode != 0:
            stderr = p.stderr.decode().strip()
            logger.error(f"Piper error: {stderr}")
            # Try to run with no arguments to see available voices
            debug = subprocess.run(["piper", "--help"], capture_output=True)
            logger.info(f"Piper help: {debug.stdout.decode()}")
            raise HTTPException(500, f"TTS error: {stderr}")
        
        # Convert WAV to MP3
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3_temp:
            mp3_path = mp3_temp.name
        
        ff = subprocess.run(
            ["ffmpeg", "-i", wav_path, "-f", "mp3", mp3_path, "-y", "-loglevel", "error"],
            capture_output=True,
            timeout=30
        )
        
        if ff.returncode != 0:
            logger.error(f"FFmpeg error: {ff.stderr.decode().strip()}")
            raise HTTPException(500, f"Audio conversion error: {ff.stderr.decode().strip()}")
        
        # Read the generated MP3 file
        with open(mp3_path, "rb") as mp3_file:
            audio_data = mp3_file.read()
            
        # Clean up temporary files
        try:
            os.unlink(wav_path)
            os.unlink(mp3_path)
        except Exception as e:
            logger.warning(f"Error cleaning up temp files: {str(e)}")
            
        return audio_data
    except subprocess.TimeoutExpired:
        logger.error("TTS process timed out")
        raise HTTPException(504, "TTS process timed out")
    except Exception as e:
        logger.error(f"TTS error: {str(e)}")
        raise HTTPException(500, f"TTS error: {str(e)}")

def elevenlabs_tts(text: str) -> bytes:
    if not ELEVENLABS_AVAILABLE:
        raise HTTPException(500, "ElevenLabs SDK not installed or key missing")
    try:
        resp = generate(text=text, voice="Rachel", model="eleven_monolingual_v1", stream=False)
        return resp.read()
    except Exception as e:
        logger.error(f"ElevenLabs error: {str(e)}")
        raise HTTPException(500, f"ElevenLabs TTS error: {str(e)}")

def openai_tts(text: str) -> bytes:
    if not OPENAI_AVAILABLE:
        raise HTTPException(500, "OpenAI SDK not installed or key missing")
    try:
        resp = openai.Audio.create(
            model="tts-1", voice="alloy", input=text, response_format="b64_json"
        )
        b64 = resp.get("audio")
        if not b64:
            raise HTTPException(500, "Bad response from OpenAI TTS")
        return base64.b64decode(b64)
    except Exception as e:
        logger.error(f"OpenAI TTS error: {str(e)}")
        raise HTTPException(500, f"OpenAI TTS error: {str(e)}")

@app.post("/process")
async def process(audio: UploadFile = File(...)):
    if not audio:
        raise HTTPException(400, "No audio file uploaded")
        
    tmp_dir = tempfile.mkdtemp()
    webm_path = os.path.join(tmp_dir, "input.webm")
    wav_path = os.path.join(tmp_dir, "input.wav")
    
    try:
        with open(webm_path, "wb") as f:
            f.write(await audio.read())
            
        ff = subprocess.run(
            ["ffmpeg", "-y", "-i", webm_path, "-ac", "1", "-ar", "16000", wav_path],
            capture_output=True
        )
        if ff.returncode != 0:
            error_msg = ff.stderr.decode().strip()
            logger.error(f"ffmpeg error: {error_msg}")
            raise HTTPException(500, f"ffmpeg error: {error_msg}")
            
        try:
            transcript = await transcribe_audio(wav_path)
            logger.info(f"Transcribed text: {transcript}")
            
            if not transcript or transcript.strip() == "":
                logger.warning("Empty transcript from Whisper")
                raise HTTPException(400, "Could not understand audio. Please try again.")
                
            response_text = await chat_with_ollama(transcript)
            
            if not response_text or response_text.strip() == "":
                logger.warning("Empty response from Ollama")
                response_text = "I'm sorry, I couldn't generate a proper response. Please try again."
                
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            raise HTTPException(500, str(e))

        # Select TTS backend and corresponding voice name
        backend = os.getenv("TTS_BACKEND", "local").lower()
        audio_bytes = None
        voice_name = "Unknown"
        
        # Try the selected backend first
        try:
            if backend == "local":
                voice_name = os.getenv("PIPER_VOICE", "en-us-ryan-low")
                audio_bytes = local_tts(response_text)
            elif backend == "elevenlabs":
                voice_name = "Rachel"
                audio_bytes = elevenlabs_tts(response_text)
            elif backend == "openai":
                voice_name = "alloy"
                audio_bytes = openai_tts(response_text)
            else:
                logger.error(f"Unknown TTS_BACKEND: {backend}")
                raise ValueError(f"Unknown TTS_BACKEND: {backend}")
        except Exception as tts_error:
            # If there's an error, log it and try a simple fallback
            logger.error(f"Error with {backend} TTS: {str(tts_error)}")
            
            # Fallback to a simple wav generation if all else fails
            try:
                logger.info("Falling back to simple audio generation")
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fallback_wav:
                    fallback_path = fallback_wav.name
                
                # Generate a simple beep sound as fallback
                simple_wav = subprocess.run(
                    ["ffmpeg", "-f", "lavfi", "-i", "sine=frequency=1000:duration=1", 
                     "-ar", "44100", "-ac", "1", fallback_path, "-y"],
                    capture_output=True
                )
                
                with open(fallback_path, "rb") as f:
                    audio_bytes = f.read()
                    
                os.unlink(fallback_path)
                voice_name = "Fallback"
            except Exception as fallback_error:
                logger.error(f"Fallback audio generation failed: {str(fallback_error)}")
                # Last resort - return empty audio
                audio_bytes = b''
                voice_name = "None"

        # Make sure we have some audio bytes to return
        if not audio_bytes:
            logger.warning("No audio data generated")
            audio_bytes = b''

        b64 = base64.b64encode(audio_bytes).decode("utf-8")
        return JSONResponse({
            "transcript": transcript,
            "response_text": response_text,
            "voice_name": voice_name,
            "audio_format": "mp3",
            "audio_base64": b64
        })
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(500, f"Server error: {str(e)}")
    finally:
        # Cleanup temp files
        try:
            if os.path.exists(webm_path):
                os.remove(webm_path)
            if os.path.exists(wav_path):
                os.remove(wav_path)
            os.rmdir(tmp_dir)
        except Exception as e:
            logger.warning(f"Cleanup error: {str(e)}")
