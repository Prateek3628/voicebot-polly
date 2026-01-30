#!/usr/bin/env python3
"""
Streaming Server with Parallel Processing Pipeline.

Features:
- Real-time speech-to-text processing (Whisper)
- Parallel intent classification + RAG retrieval
- Background TTS generation with streaming
- WebSocket for real-time communication
- Target: 2-3 second total response time

Architecture:
    Speech Input → STT → [PARALLEL] → Intent + RAG → Response → TTS Streaming
                              ↓                           ↓
                         ~0.5-1s                    Stream audio chunks
"""

import asyncio
import logging
import os
import sys
import json
import time
import uuid
import base64
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
from io import BytesIO

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

import socketio
from aiohttp import web
from openai import OpenAI

from chatbot_async import AsyncChatBot, get_async_chatbot
from tts_streaming import (
    get_tts_streaming, 
    start_tts_background, 
    get_tts_status, 
    get_audio_file,
    AUDIO_DIR
)
from config import config
from agent import ContactFormState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Socket.IO server with CORS
sio = socketio.AsyncServer(
    cors_allowed_origins='*',
    async_mode='aiohttp',
    logger=False,
    engineio_logger=False,
    ping_timeout=60,  # 60 seconds before disconnect
    ping_interval=25  # Send ping every 25 seconds
)

app = web.Application()
sio.attach(app)

# Client session storage
# {sid: {"chatbot": AsyncChatBot, "session_id": str, "interim_cache": dict, "voice_id": str}}
clients: Dict[str, Dict[str, Any]] = {}

# TTS streaming instance
tts = None

# OpenAI client for Whisper
openai_client = None

# Available Polly voices
POLLY_VOICES = {
    'male': ['Matthew', 'Joey', 'Justin', 'Kevin', 'Stephen'],
    'female': ['Salli', 'Joanna', 'Kendra', 'Kimberly', 'Ivy', 'Ruth']
}


def get_openai_client():
    """Get or create OpenAI client."""
    global openai_client
    if openai_client is None:
        openai_client = OpenAI(api_key=config.openai_api_key)
    return openai_client


def get_tts():
    """Get or create TTS instance."""
    global tts
    if tts is None:
        tts = get_tts_streaming()
    return tts


# =============================================================================
# SOCKET.IO EVENT HANDLERS
# =============================================================================

@sio.event
async def connect(sid, environ, auth=None):
    """Handle client connection with optional session restoration."""
    logger.info(f"🔗 Client {sid} connected")
    
    try:
        chatbot = get_async_chatbot()
        
        # Check if client wants to restore a previous session
        existing_session_id = auth.get('session_id') if auth else None
        
        if existing_session_id and chatbot.session_manager.is_session_valid(existing_session_id):
            # Restore existing session
            session_id = existing_session_id
            logger.info(f"♻️  Restoring session {session_id[:8]}... for {sid}")
            
            # Update session activity
            chatbot.session_manager.update_session_activity(session_id)
            
            clients[sid] = {
                'chatbot': chatbot,
                'session_id': session_id,
                'interim_cache': {},
                'last_interim_time': 0,
                'voice_id': config.polly_voice_id or 'Salli'
            }
            
            await sio.emit('status', {
                'message': 'Reconnected - session restored',
                'type': 'success',
                'session_id': session_id,
                'session_restored': True
            }, room=sid)
            
            logger.info(f"✅ Session {session_id[:8]}... restored for {sid}")
        else:
            # Create new session
            session_id, welcome = chatbot.start_session()
            
            clients[sid] = {
                'chatbot': chatbot,
                'session_id': session_id,
                'interim_cache': {},
                'last_interim_time': 0,
                'voice_id': config.polly_voice_id or 'Salli'
            }
            
            await sio.emit('status', {
                'message': 'Connected to TechGropse Streaming Server',
                'type': 'success',
                'session_id': session_id,
                'session_restored': False
            }, room=sid)
            
            # Send initial welcome message with TTS
            await send_response_with_tts(sid, welcome, show_chatbox=True, current_field='name')
            
            logger.info(f"✅ Session {session_id[:8]}... created for {sid}")
        
    except Exception as e:
        logger.error(f"Connection error for {sid}: {e}")
        await sio.emit('error', {'message': str(e)}, room=sid)


@sio.event
async def disconnect(sid):
    """Handle client disconnection - keep session alive for reconnection."""
    logger.info(f"🔌 Client {sid} disconnected")
    
    if sid in clients:
        session_id = clients[sid].get('session_id', 'unknown')
        # DON'T end the session - keep it alive in case client reconnects
        # Session will expire naturally based on session_timeout in SessionManager
        logger.info(f"📦 Session {session_id[:8]}... preserved (will expire after timeout)")
        del clients[sid]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

async def transcribe_audio(audio_base64: str, audio_format: str = 'webm') -> str:
    """
    Transcribe audio using OpenAI Whisper.
    Target: < 1 second
    """
    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(audio_base64)
        
        # Create temp file for Whisper
        with tempfile.NamedTemporaryFile(suffix=f'.{audio_format}', delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name
        
        try:
            # Transcribe with Whisper
            client = get_openai_client()
            with open(temp_path, 'rb') as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            return transcript.strip()
        finally:
            # Clean up temp file
            os.unlink(temp_path)
            
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise


async def send_response_with_tts(
    sid: str, 
    message: str, 
    show_chatbox: bool = False,
    current_field: str = None
):
    """
    Send text response IMMEDIATELY, then stream TTS audio in background.
    This ensures the user sees the response instantly while audio generates.
    """
    try:
        # Send text response FIRST (immediate - no waiting)
        await sio.emit('text_response', {
            'message': message,
            'show_chatbox': show_chatbox,
            'current_field': current_field
        }, room=sid)
        
        # Start TTS audio generation in background (non-blocking)
        if message and len(message) > 0:
            # Fire and forget - don't await
            asyncio.create_task(stream_tts_audio(sid, message))
            
    except Exception as e:
        logger.error(f"Send response error for {sid}: {e}")


async def stream_tts_audio(sid: str, text: str):
    """
    Generate TTS and stream audio chunks to client.
    Runs in background - doesn't block text response.
    Uses async TTS for better performance.
    """
    try:
        tts_start = time.time()
        
        # Signal audio start
        await sio.emit('audio_start', {}, room=sid)
        
        # Get voice for this client
        voice_id = clients.get(sid, {}).get('voice_id', 'Salli')
        
        # Generate TTS audio using async method (truly non-blocking)
        tts_instance = get_tts()
        audio_bytes = await tts_instance.synthesize_async(text)
        
        tts_time = time.time() - tts_start
        
        if audio_bytes:
            # Send audio as base64 chunk
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            await sio.emit('audio_chunk', {'data': audio_base64}, room=sid)
            logger.info(f"🔊 [{sid[:8]}] TTS: {len(audio_bytes)} bytes in {tts_time:.2f}s")
        
        # Signal audio end
        await sio.emit('audio_end', {}, room=sid)
        
    except Exception as e:
        logger.error(f"TTS streaming error for {sid}: {e}")
        await sio.emit('audio_end', {'error': str(e)}, room=sid)


def is_collecting_info(session_id: str) -> tuple:
    """
    Check if we're in the contact form collection flow.
    Returns (is_collecting, current_field)
    """
    from session_manager import session_manager
    
    state = session_manager.get_contact_form_state(session_id)
    
    if state in [
        ContactFormState.INITIAL_COLLECTING_NAME.value,
        'initial_collecting_name'
    ]:
        return True, 'name'
    elif state in [
        ContactFormState.INITIAL_COLLECTING_EMAIL.value,
        'initial_collecting_email'
    ]:
        return True, 'email'
    elif state in [
        ContactFormState.INITIAL_COLLECTING_PHONE.value,
        'initial_collecting_phone'
    ]:
        return True, 'phone'
    elif state in [
        ContactFormState.COLLECTING_DATETIME.value,
        'collecting_datetime'
    ]:
        return True, 'datetime'
    elif state in [
        ContactFormState.ASKING_CONSENT.value,
        'asking_consent'
    ]:
        return True, 'consent'
    
    return False, None


# =============================================================================
# VOICE INPUT EVENTS
# =============================================================================

@sio.event
async def voice_input(sid, data):
    """
    Handle voice input from client.
    1. Transcribe audio with Whisper
    2. Process with parallel pipeline
    3. Stream TTS response
    
    Target: 2-3 seconds total
    """
    if sid not in clients:
        await sio.emit('error', {'message': 'Session not found'}, room=sid)
        return
    
    total_start = time.time()
    
    try:
        audio_base64 = data.get('audio', '')
        audio_format = data.get('format', 'webm')
        
        if not audio_base64:
            await sio.emit('error', {'message': 'No audio data'}, room=sid)
            return
        
        logger.info(f"🎤 [{sid[:8]}] Voice input received")
        
        # Step 1: Transcribe audio
        await sio.emit('transcription_start', {}, room=sid)
        
        stt_start = time.time()
        transcription = await transcribe_audio(audio_base64, audio_format)
        stt_time = time.time() - stt_start
        
        if not transcription or len(transcription.strip()) == 0:
            await sio.emit('error', {'message': 'Could not understand audio'}, room=sid)
            return
        
        logger.info(f"📝 [{sid[:8]}] Transcribed: '{transcription}' ({stt_time:.2f}s)")
        
        await sio.emit('transcription_complete', {'text': transcription}, room=sid)
        
        # Send immediate acknowledgment for queries (to feel more responsive)
        chatbot = clients[sid]['chatbot']
        session_id = clients[sid]['session_id']
        
        # Check if we're in contact form flow or regular chat
        form_state = chatbot.session_manager.get_contact_form_state(session_id)
        in_contact_form = form_state and form_state != 'idle'
        
        # Step 2: Process with parallel pipeline
        process_start = time.time()
        result = await chatbot.process_message_async(
            user_input=transcription,
            session_id=session_id
        )
        process_time = time.time() - process_start
        
        response_text = result.get('response', '')
        intent = result.get('intent', '')
        timing_info = result.get('timing', {})
        
        # Check if we're collecting user info
        show_chatbox, current_field = is_collecting_info(session_id)
        
        # DEBUG: Log the state check
        from session_manager import session_manager
        actual_state = session_manager.get_contact_form_state(session_id)
        logger.info(f"🔍 [{sid[:8]}] State check: actual_state='{actual_state}', show_chatbox={show_chatbox}, current_field='{current_field}', intent='{intent}'")
        
        total_time = time.time() - total_start
        logger.info(f"✅ [{sid[:8]}] Total: {total_time:.2f}s (STT: {stt_time:.2f}s, Process: {process_time:.2f}s)")
        logger.info(f"   Breakdown - Parallel: {timing_info.get('parallel', 0):.2f}s, Response: {timing_info.get('response', 0):.2f}s")
        
        # Step 3: Send response with TTS
        await send_response_with_tts(
            sid, 
            response_text, 
            show_chatbox=show_chatbox,
            current_field=current_field
        )
        
    except Exception as e:
        logger.error(f"Voice input error for {sid}: {e}")
        import traceback
        traceback.print_exc()
        await sio.emit('error', {'message': f'Processing error: {str(e)}'}, room=sid)


@sio.event
async def text_query(sid, data):
    """
    Handle text input from client (chatbox).
    """
    if sid not in clients:
        await sio.emit('error', {'message': 'Session not found'}, room=sid)
        return
    
    total_start = time.time()
    
    try:
        text = data.get('text', '') if isinstance(data, dict) else str(data)
        
        if not text or not text.strip():
            await sio.emit('error', {'message': 'Empty input'}, room=sid)
            return
        
        logger.info(f"💬 [{sid[:8]}] Text: '{text}'")
        
        chatbot = clients[sid]['chatbot']
        session_id = clients[sid]['session_id']
        
        # Process with parallel pipeline
        result = await chatbot.process_message_async(
            user_input=text,
            session_id=session_id
        )
        
        response_text = result.get('response', '')
        
        # Check if we're collecting user info
        show_chatbox, current_field = is_collecting_info(session_id)
        
        total_time = time.time() - total_start
        logger.info(f"✅ [{sid[:8]}] Response in {total_time:.2f}s")
        
        # Send response with TTS
        await send_response_with_tts(
            sid, 
            response_text, 
            show_chatbox=show_chatbox,
            current_field=current_field
        )
        
    except Exception as e:
        logger.error(f"Text query error for {sid}: {e}")
        await sio.emit('error', {'message': f'Processing error: {str(e)}'}, room=sid)


# =============================================================================
# VOICE SELECTION EVENTS
# =============================================================================

@sio.event
async def get_voices(sid, data=None):
    """Send available TTS voices to client."""
    await sio.emit('available_voices', {'voices': POLLY_VOICES}, room=sid)


@sio.event
async def change_voice(sid, data):
    """Change TTS voice for client."""
    if sid not in clients:
        return
    
    voice_id = data.get('voice_id', 'Salli')
    
    # Validate voice
    all_voices = POLLY_VOICES['male'] + POLLY_VOICES['female']
    if voice_id not in all_voices:
        voice_id = 'Salli'
    
    clients[sid]['voice_id'] = voice_id
    
    logger.info(f"🔊 [{sid[:8]}] Voice changed to {voice_id}")
    
    await sio.emit('voice_changed', {'voice_id': voice_id}, room=sid)


# =============================================================================
# INTERIM SPEECH (for predictive processing)
# =============================================================================

@sio.event
async def interim_speech(sid, data):
    """
    Handle interim (partial) speech transcription.
    Runs predictive intent analysis + RAG prefetch.
    
    Target: < 500ms for quick predictions
    """
    if sid not in clients:
        return
    
    try:
        partial_text = data.get('text', '') if isinstance(data, dict) else str(data)
        
        if not partial_text or len(partial_text) < 5:
            return
        
        # Rate limit interim processing (max every 1 second)
        now = time.time()
        if now - clients[sid].get('last_interim_time', 0) < 1.0:
            return
        clients[sid]['last_interim_time'] = now
        
        chatbot = clients[sid]['chatbot']
        session_id = clients[sid]['session_id']
        
        # Process interim speech (predictive)
        result = await chatbot.process_interim_async(partial_text, session_id)
        
        # Cache interim result for potential use
        clients[sid]['interim_cache'] = result
        
        # Send interim prediction to client
        await sio.emit('interim_prediction', {
            'type': 'interim',
            'intent': result.get('intent', 'unknown'),
            'partial_text': partial_text,
            'timing': result.get('timing', 0)
        }, room=sid)
        
    except Exception as e:
        logger.error(f"Interim processing error for {sid}: {e}")


# =============================================================================
# ALIASES AND UTILITY EVENTS
# =============================================================================

@sio.event
async def query(sid, data):
    """Alias for text_query (compatibility)."""
    await text_query(sid, data)


@sio.event
async def user_query(sid, data):
    """Alias for text_query (compatibility)."""
    await text_query(sid, data)


@sio.event
async def check_audio_status(sid, data):
    """Check TTS audio generation status."""
    try:
        audio_id = data.get('audio_id') if isinstance(data, dict) else str(data)
        
        if not audio_id:
            await sio.emit('error', {'message': 'Missing audio_id'}, room=sid)
            return
        
        status = get_tts_status(audio_id)
        await sio.emit('audio_status', status, room=sid)
        
    except Exception as e:
        logger.error(f"Audio status check error: {e}")
        await sio.emit('error', {'message': str(e)}, room=sid)


@sio.event
async def health_check(sid, data=None):
    """Health check event."""
    await sio.emit('health_status', {
        'status': 'ok',
        'server': 'streaming',
        'tts_provider': get_tts().provider_name if tts else 'not_initialized'
    }, room=sid)


# =============================================================================
# HTTP ENDPOINTS (for audio file serving and REST API)
# =============================================================================

async def http_health(request):
    """HTTP health check endpoint."""
    return web.json_response({
        'status': 'ok',
        'server': 'streaming',
        'endpoints': [
            'WS: connect, interim_speech, final_speech, query',
            'HTTP: /health, /audio/<id>, /audio/<id>/status, /api/query'
        ]
    })


async def http_audio_status(request):
    """HTTP endpoint for audio status check."""
    audio_id = request.match_info.get('audio_id')
    
    if not audio_id:
        return web.json_response({'error': 'Missing audio_id'}, status=400)
    
    status = get_tts_status(audio_id)
    return web.json_response(status)


async def http_audio_file(request):
    """HTTP endpoint to serve audio file."""
    audio_id = request.match_info.get('audio_id')
    
    if not audio_id:
        return web.json_response({'error': 'Missing audio_id'}, status=400)
    
    status = get_tts_status(audio_id)
    
    if status['status'] == 'not_found':
        return web.json_response({'error': 'Audio not found'}, status=404)
    
    if status['status'] == 'pending':
        return web.json_response({'status': 'pending', 'message': 'Audio still generating'}, status=202)
    
    if status['status'] == 'error':
        return web.json_response({'error': status.get('error')}, status=500)
    
    # Serve the audio file
    audio_bytes = get_audio_file(audio_id)
    if not audio_bytes:
        return web.json_response({'error': 'Audio file not found'}, status=404)
    
    return web.Response(
        body=audio_bytes,
        content_type='audio/mpeg',
        headers={
            'Content-Disposition': f'inline; filename="{audio_id}.mp3"',
            'Access-Control-Allow-Origin': '*'
        }
    )


async def http_api_query(request):
    """
    REST API endpoint for queries.
    POST /api/query {"message": "...", "session_id": "...", "fast_mode": false}
    """
    try:
        data = await request.json()
        message = data.get('message', '')
        session_id = data.get('session_id')
        fast_mode = data.get('fast_mode', False)
        
        if not message:
            return web.json_response({'error': 'Missing message'}, status=400)
        
        # Create temporary session if not provided
        chatbot = get_async_chatbot()
        if not session_id:
            session_id, _ = chatbot.start_session()
        
        # Process query
        result = await chatbot.process_message_async(
            user_input=message,
            session_id=session_id,
            fast_mode=fast_mode
        )
        
        # Start TTS if response exists
        audio_id = None
        response_text = result.get('response', '')
        if response_text:
            try:
                audio_id = start_tts_background(response_text)
            except Exception:
                pass
        
        return web.json_response({
            **result,
            'audio_id': audio_id
        })
        
    except Exception as e:
        logger.error(f"API query error: {e}")
        return web.json_response({'error': str(e)}, status=500)


# Add HTTP routes
app.router.add_get('/', http_health)
app.router.add_get('/health', http_health)
app.router.add_get('/audio/{audio_id}/status', http_audio_status)
app.router.add_get('/audio/{audio_id}', http_audio_file)
app.router.add_post('/api/query', http_api_query)

# Serve static files
static_path = Path(__file__).parent / 'static'
if static_path.exists():
    app.router.add_static('/static/', path=str(static_path), name='static')


# =============================================================================
# SERVER STARTUP
# =============================================================================

def check_environment():
    """Check required environment variables."""
    issues = []
    
    if not config.openai_api_key:
        issues.append("OPENAI_API_KEY not set")
    
    if not config.aws_access_key_id or not config.aws_secret_access_key:
        logger.warning("AWS credentials not set - TTS may not work")
    
    return issues


def main(host='0.0.0.0', port=5001):
    """Start the streaming server."""
    print("=" * 60)
    print("    🚀 TECHGROPSE STREAMING SERVER (Parallel Processing)")
    print("=" * 60)
    print(f"\n✅ Server: http://{host}:{port}")
    print(f"✅ WebSocket: ws://{host}:{port}")
    print(f"\n🎤 Voice Interface:")
    print(f"   http://localhost:{port}/static/voice_to_voice.html")
    print("\n📡 Socket.IO Events:")
    print("   • connect - Start session")
    print("   • voice_input - Audio from microphone")
    print("   • text_query - Text from chatbox")
    print("   • get_voices / change_voice - Voice selection")
    print("\n🌐 HTTP Endpoints:")
    print("   • GET  /health - Health check")
    print("   • GET  /audio/<id> - Download audio")
    print("   • POST /api/query - REST API query")
    print("\n⚡ Target Response Time: 2-3 seconds")
    print("-" * 60)
    
    web.run_app(app, host=host, port=port)


if __name__ == '__main__':
    try:
        # Check environment
        issues = check_environment()
        if issues:
            print("⚠️ Environment warnings:")
            for issue in issues:
                print(f"   • {issue}")
        
        # Initialize TTS
        try:
            get_tts()
            print("✅ TTS provider initialized")
        except Exception as e:
            print(f"⚠️ TTS initialization failed: {e}")
        
        # Pre-load ML models to avoid delay on first query
        print("🔄 Pre-loading ML models (this may take a few seconds)...")
        try:
            from vectorstore.chromadb_client import get_chromadb_client
            from utils.reranker import get_reranker
            from agent_async import get_async_agent
            
            # Load ChromaDB client (includes embedding model)
            chromadb = get_chromadb_client()
            print("   ✅ Embedding model loaded")
            
            # Load reranker (cross-encoder model)
            reranker = get_reranker()
            # Force model loading by accessing the model property
            _ = reranker.model
            print("   ✅ Reranker model loaded")
            
            # Pre-initialize the async agent singleton
            agent = get_async_agent()
            print("   ✅ Async agent initialized")
            
            print("✅ All ML models pre-loaded - ready for fast queries!")
        except Exception as e:
            print(f"⚠️ Model pre-loading failed: {e}")
            print("   Models will load on first query (may cause initial delay)")
        
        # Start server
        main(host='0.0.0.0', port=5001)
        
    except KeyboardInterrupt:
        print("\n👋 Server shutting down...")
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)
