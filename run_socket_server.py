#!/usr/bin/env python3
"""
Simple script to run the raw socket server with proper environment setup.
"""
import os
import sys
from pathlib import Path

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

def main():
    """Run the socket server."""
    try:
        # Import and run the socket server
        from socket_server import main as start_server
        
        print("🚀 Starting TechGropse Virtual Representative Socket.IO Server...")
        print("🔌 Socket.IO endpoint: http://51.38.38.66:5000")
        print("📡 Protocol: Socket.IO with WebSocket transport")
        print("Press Ctrl+C to stop the server\n")
        
        # Run the server
        start_server(
            host='0.0.0.0',
            port=5001
        )
        
    except KeyboardInterrupt:
        print("\n👋 Server shutting down gracefully...")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
