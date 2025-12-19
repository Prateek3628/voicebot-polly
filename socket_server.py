#!/usr/bin/env python3
"""
Socket.IO Server for TechGropse Virtual Representative
Works with frontend using socket.io-client
"""

import socketio
import logging
import asyncio
import os
import sys
from pathlib import Path
from aiohttp import web

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

from chatbot import ChatBot
from config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Socket.IO server with CORS enabled
sio = socketio.AsyncServer(
    cors_allowed_origins='*',  # Allow all origins for development
    async_mode='aiohttp',
    logger=True,
    engineio_logger=True
)

app = web.Application()
sio.attach(app)

# Store client sessions
clients = {}

@sio.event
async def connect(sid, environ, auth=None):
    """Handle client connection with optional auth data"""
    logger.info(f"🔗 Client {sid} connected")
    if auth:
        logger.info(f"📋 Connection data: {auth}")
    
    try:
        # Create individual session for this client
        chatbot = ChatBot()
        session_id = chatbot.start_session()  # Get session_id
        
        clients[sid] = {
            'chatbot': chatbot,
            'session_id': session_id  # Store separately
        }
        
        # Send greeting
        await sio.emit('status', {
            'message': 'Connected to TechGropse Virtual Representative',
            'type': 'success'
        }, room=sid)
        
        logger.info(f"✅ Session created for client {sid} with session_id: {session_id[:8]}...")
        
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error creating session for {sid}: {e}")
        
        await sio.emit('error', {
            'message': f'Failed to create session: {error_message}'
        }, room=sid)

@sio.event
async def disconnect(sid):
    """Handle client disconnection"""
    logger.info(f"🔌 Client {sid} disconnected")
    
    if sid in clients:
        client_data = clients[sid]
        chatbot = client_data['chatbot']
        session_id = client_data['session_id']
        try:
            chatbot.end_session(session_id)  # Pass session_id
        except:
            pass
        del clients[sid]
        logger.info(f"🗑️ Session cleaned up for client {sid}")

@sio.event
async def query(sid, data):
    """Handle query event from client (alias for user_query)"""
    await user_query(sid, data)

@sio.event
async def user_query(sid, data):
    """Handle user_query event from client"""
    try:
        if sid not in clients:
            await sio.emit('error', {
                'message': 'Session not found'
            }, room=sid)
            return
        
        # Extract query text
        if isinstance(data, dict):
            query_text = data.get('message', '') or data.get('query', '')
        else:
            query_text = str(data)
        
        if not query_text or not query_text.strip():
            await sio.emit('error', {
                'message': 'Empty query'
            }, room=sid)
            return
        
        client_data = clients[sid]
        chatbot = client_data['chatbot']
        session_id = client_data['session_id']  # Get session_id
        
        logger.info(f"💬 Client {sid}: '{query_text}'")
        
        # Send acknowledgment
        await sio.emit('query_received', {
            'message': query_text,
            'status': 'processing'
        }, room=sid)
        
        try:
            # Process query
            logger.info(f"🔄 Processing query for {sid}...")
            
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                chatbot.process_message,
                query_text,
                session_id  # Pass session_id
            )
            
            logger.info(f"✅ Client {sid} processing completed")
            
            # Send response
            await sio.emit('bot_response', {
                'message': response,
                'original_query': query_text
            }, room=sid)
            
            logger.info(f"📤 Sent to client {sid}: {response[:50]}...")
            
        except Exception as e:
            logger.error(f"Error processing query for {sid}: {e}")
            await sio.emit('error', {
                'message': f'Processing error: {str(e)}'
            }, room=sid)
            
    except Exception as e:
        logger.error(f"Error handling query for {sid}: {e}")
        await sio.emit('error', {
            'message': str(e)
        }, room=sid)

@sio.event
async def health_check(sid, data=None):
    """Handle health_check event from client"""
    try:
        if sid not in clients:
            await sio.emit('error', {
                'message': 'Session not found'
            }, room=sid)
            return
        
        chatbot = clients[sid]['chatbot']
        
        loop = asyncio.get_event_loop()
        health_status = await loop.run_in_executor(
            None,
            chatbot.health_check
        )
        
        await sio.emit('health_status', health_status, room=sid)
        
    except Exception as e:
        logger.error(f"Health check failed for {sid}: {e}")
        await sio.emit('error', {
            'message': f'Health check failed: {str(e)}'
        }, room=sid)

@sio.event
async def get_stats(sid, data=None):
    """Handle get_stats event from client"""
    try:
        if sid not in clients:
            await sio.emit('error', {
                'message': 'Session not found'
            }, room=sid)
            return
        
        chatbot = clients[sid]['chatbot']
        session_id = clients[sid]['session_id']  # Get session_id
        
        loop = asyncio.get_event_loop()
        stats = await loop.run_in_executor(
            None,
            chatbot.get_session_stats,
            session_id  # Pass session_id
        )
        
        await sio.emit('session_stats', stats, room=sid)
        
    except Exception as e:
        logger.error(f"Failed to get stats for {sid}: {e}")
        await sio.emit('error', {
            'message': f'Failed to get statistics: {str(e)}'
        }, room=sid)

# Health check endpoint
async def health(request):
    """Health check endpoint"""
    return web.Response(text='Socket.IO Server Running')

# Add routes
app.router.add_get('/', health)
app.router.add_get('/health', health)

def check_environment():
    """Check if all required environment variables are set."""
    issues = []
    
    # Check OpenAI API key
    if not config.openai_api_key:
        issues.append("OpenAI API key is not set. Please set OPENAI_API_KEY environment variable.")
    
    # Check if data file exists
    if not os.path.exists(config.data_file_path):
        issues.append(f"Data file not found: {config.data_file_path}")
    
    return issues

def main(host='0.0.0.0', port=5000):
    """Start the Socket.IO server"""
    logger.info(f"🚀 Starting Socket.IO Server on {host}:{port}")
    logger.info("� Each client gets their own session")
    logger.info("🗑️ Sessions expire when clients disconnect")
    logger.info("🌐 CORS enabled for frontend integration")
    logger.info("-" * 50)
    
    web.run_app(app, host=host, port=port)

if __name__ == '__main__':
    try:
        # Check environment
        issues = check_environment()
        if issues:
            print("❌ Environment check failed:")
            for issue in issues:
                print(f"   • {issue}")
            sys.exit(1)
        
        # Start the Socket.IO server
        main(host='0.0.0.0', port=5000)
        
    except KeyboardInterrupt:
        print("\n👋 Server shutting down...")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)
