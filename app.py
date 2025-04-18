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
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL", "purevoice-granite")
OLLAMA_HOST   = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_URL    = f"{OLLAMA_HOST}/api/generate"
OLLAMA_HEALTH_URL = f"{OLLAMA_HOST}/api/version"
OLLAMA_SYSTEM_PROMPT = os.getenv("OLLAMA_SYSTEM_PROMPT", "")
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
    title="PureVoice AI Assistant",
    description="A privacy-focused voice assistant using Whisper, Ollama, and Piper TTS",
    version="1.0.0"
)

# Add CORS middleware
# Allow local development origins and file:// (null) for development flexibility
allowed_origins = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://0.0.0.0:8000",
    "https://localvoice.lucianaferreira.pro",
    "null"
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
    # Check Ollama health
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
    
    # Check Whisper model status
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
                "backend": "local",
                "voice": os.getenv("PIPER_VOICE", "en-us-ryan-low")
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

async def test_generate():
    """Test if model can generate a response"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": "hello",
                "stream": False
            }
            response = await client.post(OLLAMA_URL, json=payload)
            if response.status_code == 200:
                logger.info(f"Model {OLLAMA_MODEL} is ready to generate")
                return True
            else:
                logger.warning(f"Model {OLLAMA_MODEL} generation test failed with status {response.status_code}")
                return False
    except Exception as e:
        logger.warning(f"Generation test error: {str(e)}")
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
        try:
            # Check API health first
            if not await check_ollama_health():
                logger.warning(f"Ollama API not healthy on attempt {attempt+1}/{MAX_RETRIES}")
                delay = min(RETRY_DELAY * (1.5 ** attempt), 3.0)
                await asyncio.sleep(delay)
                continue
                
            # Now test generation
            if await test_generate():
                logger.info("Model is successfully generating responses")
                return True
                
            # If we got here, the API is up but generate failed
            logger.warning(f"Waiting for model to be ready for generation, attempt {attempt+1}/{MAX_RETRIES}")
            delay = min(RETRY_DELAY * (1.5 ** attempt), 3.0)
            await asyncio.sleep(delay)
            
        except Exception as e:
            logger.warning(f"Error in ensure_ollama_ready: {str(e)}")
            delay = min(RETRY_DELAY * (1.5 ** attempt), 3.0)
            await asyncio.sleep(delay)
            
    logger.warning(f"Ollama model generation readiness check failed after {MAX_RETRIES} retries")
    return False

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
    
    # Ensure Ollama is ready
    if not await ensure_ollama_ready():
        raise HTTPException(503, "Language model service is not available right now. Please try again in a moment.")
    
    url = OLLAMA_URL
    logger.info(f"Using Ollama URL: {url}")
    
    # Use a more minimal payload with just what's needed
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": text,
        "stream": False
    }
    
    # Add system prompt if available
    if OLLAMA_SYSTEM_PROMPT:
        logger.info("Using custom system prompt")
        payload["system"] = OLLAMA_SYSTEM_PROMPT
    
    logger.info(f"Sending request to Ollama API using model: {OLLAMA_MODEL}")
    
    # Increase timeout for better reliability
    timeout = 10.0
            
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            res = await client.post(url, json=payload)
            res.raise_for_status()
            data = res.json()
            
            # Extract text from the /api/generate response
            if "response" in data:
                response = data["response"].strip()
                if response:
                    logger.info("Successfully got response from model")
                    return response
                else:
                    logger.warning("Received empty response from Ollama")
                    return "I'm sorry, I don't have a response for that."
            else:
                logger.warning(f"Unexpected Ollama response format: {data}")
                return "I don't understand that question. Could you try asking in a different way?"
    except httpx.RequestError as e:
        logger.error(f"Connection error when talking to Ollama: {e}")
        # propagate as a service unavailable
        raise HTTPException(503, "Language model service is not available right now. Please try again later.")
    except httpx.HTTPStatusError as e:
        logger.error(f"Ollama API returned error status: {e.response.status_code} - {e}")
        raise HTTPException(e.response.status_code, f"LLM service error: {e.response.text}")
    except Exception as e:
        logger.error(f"Unexpected error in Ollama API call: {e}")
        raise HTTPException(500, "Internal error while querying language model.")

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

@app.post("/process")
async def process(audio: UploadFile = File(...)):
    if not audio:
        raise HTTPException(400, "No audio file uploaded")
        
    if not model:
        raise HTTPException(503, "Speech recognition model not available")
        
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
            raise HTTPException(500, f"Audio conversion error: {error_msg}")
            
        # Transcribe the audio
        transcript = await transcribe_audio(wav_path)
        logger.info(f"Transcribed text: {transcript}")
        
        if not transcript or transcript.strip() == "":
            logger.warning("Empty transcript from Whisper")
            raise HTTPException(400, "Could not understand audio. Please try again.")
            
        # Get response from the language model
        response_text = await chat_with_ollama(transcript)
        
        if not response_text or response_text.strip() == "":
            logger.warning("Empty response from Ollama")
            raise HTTPException(500, "The language model returned an empty response. Please try again.")
        
        # Generate speech from the response text
        voice_name = os.getenv("PIPER_VOICE", "en-us-ryan-low")
        audio_bytes = local_tts(response_text)

        # Encode audio as base64
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
