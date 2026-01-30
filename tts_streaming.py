"""
TTS Streaming with Background Audio Generation.
Supports AWS Polly and Eleven Labs with async generation and status polling.
"""
import asyncio
import logging
import os
import uuid
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import boto3
import requests

from config import config

logger = logging.getLogger(__name__)

# Audio storage directory
AUDIO_DIR = Path("audio_cache")
AUDIO_DIR.mkdir(exist_ok=True)

# Thread pool for TTS operations
_tts_executor = ThreadPoolExecutor(max_workers=2)

# Audio generation status tracking
# {audio_id: {"status": "pending|ready|error", "path": str, "error": str, "start_time": float}}
_audio_status: Dict[str, Dict[str, Any]] = {}


class TTSProvider:
    """Base TTS provider interface."""
    
    def synthesize(self, text: str) -> bytes:
        raise NotImplementedError


class PollyProvider(TTSProvider):
    """AWS Polly TTS Provider."""
    
    def __init__(self):
        self.client = boto3.client(
            'polly',
            aws_access_key_id=config.aws_access_key_id,
            aws_secret_access_key=config.aws_secret_access_key,
            region_name=config.aws_region
        )
        self.voice_id = config.polly_voice_id or 'Salli'
        self.output_format = config.polly_output_format or 'mp3'
        logger.info(f"Polly provider initialized: voice={self.voice_id}")
    
    def synthesize(self, text: str) -> bytes:
        """Synthesize speech using AWS Polly."""
        try:
            response = self.client.synthesize_speech(
                Text=text,
                OutputFormat=self.output_format,
                VoiceId=self.voice_id,
                Engine='neural'
            )
            return response['AudioStream'].read()
        except Exception as e:
            logger.error(f"Polly synthesis error: {e}")
            raise


class ElevenLabsProvider(TTSProvider):
    """Eleven Labs TTS Provider."""
    
    def __init__(self):
        self.api_key = os.getenv('ELEVEN_LABS_API_KEY')
        self.voice_id = os.getenv('ELEVEN_LABS_VOICE_ID', 'JBFqnCBsd6RMkjVDRZzb')
        self.model_id = os.getenv('ELEVEN_LABS_MODEL_ID', 'eleven_turbo_v2')
        
        if not self.api_key:
            raise ValueError("ELEVEN_LABS_API_KEY not configured")
        
        logger.info(f"ElevenLabs provider initialized: voice={self.voice_id}")
    
    def synthesize(self, text: str) -> bytes:
        """Synthesize speech using Eleven Labs."""
        try:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
            headers = {
                'xi-api-key': self.api_key,
                'Content-Type': 'application/json',
                'Accept': 'audio/mpeg'
            }
            payload = {
                "text": text,
                "model_id": self.model_id,
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75,
                    "style": 0.0,
                    "use_speaker_boost": True
                }
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code != 200:
                raise Exception(f"ElevenLabs API error: {response.status_code}")
            
            return response.content
            
        except Exception as e:
            logger.error(f"ElevenLabs synthesis error: {e}")
            raise


class TTSStreaming:
    """
    TTS Streaming Manager with background generation.
    
    Features:
    - Background audio generation (non-blocking)
    - Status polling for completion
    - Audio file caching
    - Support for Polly and ElevenLabs
    """
    
    def __init__(self, provider: str = None):
        """
        Initialize TTS streaming manager.
        
        Args:
            provider: "polly" or "elevenlabs" (default from env)
        """
        provider = provider or os.getenv('TTS_PROVIDER', 'polly').lower()
        
        if provider in ('elevenlabs', 'eleven'):
            try:
                self.provider = ElevenLabsProvider()
                self.provider_name = 'elevenlabs'
            except ValueError:
                logger.warning("ElevenLabs not configured, falling back to Polly")
                self.provider = PollyProvider()
                self.provider_name = 'polly'
        else:
            self.provider = PollyProvider()
            self.provider_name = 'polly'
        
        logger.info(f"TTS Streaming initialized with {self.provider_name}")
    
    def start_generation(self, text: str) -> str:
        """
        Start background TTS generation.
        Returns immediately with audio_id for status polling.
        
        Args:
            text: Text to synthesize
            
        Returns:
            audio_id: Unique identifier for status polling
        """
        audio_id = str(uuid.uuid4())
        
        _audio_status[audio_id] = {
            "status": "pending",
            "path": None,
            "error": None,
            "start_time": time.time(),
            "text_length": len(text)
        }
        
        # Start background thread for generation
        thread = threading.Thread(
            target=self._generate_audio_background,
            args=(text, audio_id),
            daemon=True
        )
        thread.start()
        
        logger.info(f"Started background TTS: audio_id={audio_id}, text_length={len(text)}")
        
        return audio_id
    
    def _generate_audio_background(self, text: str, audio_id: str):
        """Background thread function for audio generation."""
        try:
            start = time.time()
            
            # Generate audio
            audio_bytes = self.provider.synthesize(text)
            
            # Save to file
            audio_path = AUDIO_DIR / f"{audio_id}.mp3"
            with open(audio_path, 'wb') as f:
                f.write(audio_bytes)
            
            elapsed = time.time() - start
            
            _audio_status[audio_id] = {
                "status": "ready",
                "path": str(audio_path),
                "error": None,
                "generation_time": elapsed,
                "file_size": len(audio_bytes)
            }
            
            logger.info(f"TTS complete: audio_id={audio_id}, time={elapsed:.2f}s, size={len(audio_bytes)}")
            
        except Exception as e:
            logger.error(f"Background TTS error for {audio_id}: {e}")
            _audio_status[audio_id] = {
                "status": "error",
                "path": None,
                "error": str(e)
            }
    
    async def start_generation_async(self, text: str) -> str:
        """
        Async version of start_generation.
        Returns immediately with audio_id.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _tts_executor,
            self.start_generation,
            text
        )
    
    def get_status(self, audio_id: str) -> Dict[str, Any]:
        """
        Get generation status for an audio_id.
        
        Returns:
            Dict with status, path (if ready), error (if failed)
        """
        status = _audio_status.get(audio_id)
        
        if not status:
            return {"status": "not_found", "error": "Audio ID not found"}
        
        return {
            "audio_id": audio_id,
            "status": status["status"],
            "path": status.get("path"),
            "error": status.get("error"),
            "generation_time": status.get("generation_time")
        }
    
    def get_audio_bytes(self, audio_id: str) -> Optional[bytes]:
        """
        Get audio bytes if generation is complete.
        
        Returns:
            Audio bytes or None if not ready
        """
        status = _audio_status.get(audio_id)
        
        if not status or status["status"] != "ready":
            return None
        
        audio_path = Path(status["path"])
        if not audio_path.exists():
            return None
        
        with open(audio_path, 'rb') as f:
            return f.read()
    
    def synthesize_sync(self, text: str) -> bytes:
        """
        Synchronous synthesis (blocking).
        Use for simple cases where immediate audio is needed.
        """
        return self.provider.synthesize(text)
    
    async def synthesize_async(self, text: str) -> bytes:
        """
        Async synthesis (still blocking internally but non-blocking to event loop).
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _tts_executor,
            self.provider.synthesize,
            text
        )
    
    async def wait_for_audio(self, audio_id: str, timeout: float = 30.0) -> Optional[bytes]:
        """
        Wait for audio generation to complete with polling.
        
        Args:
            audio_id: Audio identifier
            timeout: Maximum wait time in seconds
            
        Returns:
            Audio bytes or None if timeout/error
        """
        start = time.time()
        poll_interval = 0.1  # 100ms polling
        
        while time.time() - start < timeout:
            status = self.get_status(audio_id)
            
            if status["status"] == "ready":
                return self.get_audio_bytes(audio_id)
            elif status["status"] == "error":
                logger.error(f"Audio generation failed: {status.get('error')}")
                return None
            
            await asyncio.sleep(poll_interval)
        
        logger.warning(f"Audio generation timeout for {audio_id}")
        return None
    
    def cleanup_old_audio(self, max_age_seconds: int = 3600):
        """Clean up old audio files from cache."""
        import glob
        
        now = time.time()
        cleaned = 0
        
        for audio_file in AUDIO_DIR.glob("*.mp3"):
            try:
                file_age = now - audio_file.stat().st_mtime
                if file_age > max_age_seconds:
                    audio_file.unlink()
                    cleaned += 1
            except Exception:
                pass
        
        if cleaned:
            logger.info(f"Cleaned {cleaned} old audio files")


# Singleton instance
_tts_instance = None

def get_tts_streaming() -> TTSStreaming:
    """Get or create TTS streaming singleton."""
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = TTSStreaming()
    return _tts_instance


# Convenience functions
def start_tts_background(text: str) -> str:
    """Start background TTS and return audio_id."""
    return get_tts_streaming().start_generation(text)


def get_tts_status(audio_id: str) -> Dict[str, Any]:
    """Get TTS generation status."""
    return get_tts_streaming().get_status(audio_id)


def get_audio_file(audio_id: str) -> Optional[bytes]:
    """Get audio file bytes if ready."""
    return get_tts_streaming().get_audio_bytes(audio_id)
