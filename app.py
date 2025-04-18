import os
import uuid
import base64
import tempfile
import subprocess
import logging
import time
import asyncio
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
OLLAMA_HOST   = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_URL    = f"{OLLAMA_HOST}/api/generate"
OLLAMA_HEALTH_URL = f"{OLLAMA_HOST}/api/version"
MAX_RETRIES   = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY   = float(os.getenv("RETRY_DELAY", "0.5"))

# Common questions and answers for fast responses
COMMON_ANSWERS = {
    "capital of the united states": "The capital of the United States is Washington, D.C.",
    "color of the sky": "The sky appears blue during the day due to a phenomenon called Rayleigh scattering, where the atmosphere scatters blue light from the sun more than other colors.",
    "us president": "The current President of the United States is Joe Biden, who took office on January 20, 2021."
}

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
    # Check Ollama health with better diagnostics
    try:
        ollama_status = await check_ollama_health()
        # Check if model is loaded
        model_loaded = False
        
        if ollama_status:
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    response = await client.post(
                        f"{OLLAMA_HOST}/api/tags", 
                        json={},
                        timeout=2.0
                    )
                    if response.status_code == 200:
                        model_data = response.json()
                        if "models" in model_data:
                            for model_info in model_data.get("models", []):
                                if model_info.get("name", "").startswith(OLLAMA_MODEL.split(":")[0]):
                                    model_loaded = True
                                    logger.info(f"Found model {model_info.get('name')} matching {OLLAMA_MODEL}")
                                    break
            except Exception as e:
                logger.warning(f"Could not verify model loading status: {str(e)}")
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        ollama_status = False
        model_loaded = False
    
    # Check Whisper model status with more details
    whisper_status = model is not None
    whisper_detail = "loaded" if whisper_status else "not loaded"
    
    # Compute overall status
    overall_status = "ok" if ollama_status and whisper_status else "degraded"
    if not ollama_status:
        overall_status = "critical"
    
    # Return detailed health information
    return {
        "status": overall_status,
        "version": "1.0.0",
        "timestamp": int(time.time()),
        "components": {
            "ollama": {
                "status": "ready" if ollama_status else "unavailable", 
                "model": OLLAMA_MODEL,
                "model_loaded": model_loaded,
                "url": OLLAMA_URL
            },
            "whisper": {
                "status": whisper_detail,
                "model": WHISPER_MODEL
            },
            "tts": {
                "backend": os.getenv("TTS_BACKEND", "local"),
                "voice": os.getenv("PIPER_VOICE", "en-us-ryan-low") if os.getenv("TTS_BACKEND", "local") == "local" else "default"
            }
        }
    }

async def check_ollama_health(retries=1):
    """Check if Ollama API is healthy and ready to serve requests"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(OLLAMA_HEALTH_URL)
            if response.status_code == 200:
                logger.info("Ollama API is healthy and ready")
                return True
            else:
                logger.warning(f"Ollama API returned status code {response.status_code}")
                return False
    except httpx.HTTPError as e:
        if retries > 0:
            logger.warning(f"Ollama health check failed: {str(e)}. Retrying...")
            return False
        else:
            logger.error(f"Ollama health check failed after retries: {str(e)}")
            return False

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

async def ensure_ollama_ready():
    """Ensure Ollama is ready before sending requests, with exponential backoff"""
    for attempt in range(MAX_RETRIES):
        if await check_ollama_health():
            return True
        
        # Shorter delays for faster response
        delay = min(RETRY_DELAY * (1.5 ** attempt), 3.0)
        logger.info(f"Waiting for Ollama to be ready, retrying in {delay:.1f}s (attempt {attempt+1}/{MAX_RETRIES})")
        await asyncio.sleep(delay)  # Use asyncio.sleep instead of time.sleep
    
    logger.warning("Ollama service not fully ready after maximum retries - trying anyway")
    return True  # Return True to try anyway

def find_common_answer(text: str) -> str:
    """Look for predefined answers to common questions"""
    text_lower = text.lower()
    
    # Check for specific questions using keywords
    for key, answer in COMMON_ANSWERS.items():
        if key in text_lower:
            logger.info(f"Using predefined answer for '{key}'")
            return answer
            
    return None

async def chat_with_ollama(text: str) -> str:
    # Check for common questions we can answer without the LLM
    common_answer = find_common_answer(text)
    if common_answer:
        return common_answer
    
    url = OLLAMA_URL
    logger.info(f"Using Ollama URL: {url}")
    
    # Format prompt specifically for Mistral models
    if "mistral" in OLLAMA_MODEL.lower():
        logger.info("Using Mistral format for prompt")
        if "instruct" in OLLAMA_MODEL.lower():
            formatted_prompt = f"[INST] {text} [/INST]"
        else:
            formatted_prompt = text
    else:
        formatted_prompt = text
    
    # Format for /api/generate endpoint with correct stop tokens for Mistral
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": formatted_prompt,
        "stream": False,
        "options": {
            "num_ctx": 2048,
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 512,
            "stop": ["[INST]", "[/INST]"] if "mistral" in OLLAMA_MODEL.lower() else []
        }
    }
    
    logger.info(f"Sending request to Ollama API: {url}")
    logger.info(f"Using model: {OLLAMA_MODEL}")
    logger.debug(f"Payload: {payload}")
    
    # Implement retry logic
    max_retries = int(os.getenv("MAX_RETRIES", "3"))
    retry_delay = float(os.getenv("RETRY_DELAY", "2.0"))
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            # Use a longer timeout for the first inference which can be slower
            timeout = 30.0 if attempt == 0 else 15.0
            logger.info(f"Attempt {attempt+1}/{max_retries} with timeout {timeout}s")
            
            async with httpx.AsyncClient(timeout=timeout) as client:
                # Log request details for debugging
                logger.info(f"Sending POST to {url} with model {OLLAMA_MODEL}")
                
                res = await client.post(url, json=payload)
                res.raise_for_status()
                data = res.json()
                
                # Extract text from the /api/generate response
                if "response" in data:
                    logger.info("Successfully got response from model")
                    return data["response"].strip()
                else:
                    logger.warning(f"Unexpected Ollama response format: {data}")
                    # Try to extract useful text from other fields
                    for key, value in data.items():
                        if isinstance(value, str) and len(value) > 10:
                            logger.info(f"Found alternate response in field: {key}")
                            return value.strip()
                    
                    # If we can't extract a response, raise an exception
                    raise ValueError(f"Could not extract response from Ollama API. Got: {data}")
        except Exception as e:
            last_exception = e
            logger.warning(f"Ollama API attempt {attempt+1}/{max_retries} failed: {str(e)}")
            
            if attempt < max_retries - 1:
                wait_time = retry_delay * (attempt + 1)
                logger.info(f"Waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"All {max_retries} attempts to Ollama API failed")
    
    # If we got here, all attempts failed
    logger.error(f"Ollama API error after {max_retries} attempts: {str(last_exception)}")
    
    # Provide a more informative error message for debugging
    error_msg = f"Model '{OLLAMA_MODEL}' failed to generate a response. Error: {str(last_exception)}"
    logger.error(error_msg)
    
    raise HTTPException(status_code=500, detail=error_msg)

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
