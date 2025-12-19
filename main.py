"""
Main entry point for the TechGropse Virtual Representative application.
"""
import logging
import os
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

from chatbot import ChatBot
from initialise_data import (
    load_privacy_policy_data, 
    create_document_metadata, 
    process_and_embed_documents
)
from vectorstore.chromadb_client import ChromaDBClient
from session_manager import session_manager
from config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_environment():
    """Check if all required environment variables and dependencies are set."""
    issues = []
    
    # Check OpenAI API key - try multiple sources
    openai_key = None
    try:
        # First try to load from config (which reads .env)
        from config import config
        openai_key = config.openai_api_key
    except Exception as e:
        # If config fails, try direct environment access
        openai_key = os.getenv('OPENAI_API_KEY')
    
    if not openai_key:
        issues.append("OpenAI API key is not set. Please set OPENAI_API_KEY environment variable or add it to .env file.")
    else:
        # Set it in environment for CrewAI to access
        os.environ['OPENAI_API_KEY'] = openai_key.strip('"\'')
    
    # Check if data file exists
    if not os.path.exists(config.data_file_path):
        issues.append(f"Data file not found: {config.data_file_path}")
    
    # Check Redis connection
    try:
        session_manager.redis_client.ping()
    except Exception as e:
        issues.append(f"Redis connection failed: {e}")
    
    return issues


def initialize_data_if_needed():
    """Initialize ChromaDB data if the collection is empty."""
    try:
        chromadb_client = ChromaDBClient()
        
        if chromadb_client.is_collection_empty():
            logger.info("ChromaDB collection is empty. Initializing data...")
            
            # Load privacy policy data
            content = load_privacy_policy_data(config.data_file_path)
            
            # Create metadata
            metadata = create_document_metadata(config.data_file_path)
            
            # Process and embed documents
            success = process_and_embed_documents(chromadb_client, content, metadata)
            
            if success:
                logger.info("Data initialization completed successfully")
                return True
            else:
                logger.error("Data initialization failed")
                return False
        else:
            doc_count = chromadb_client.get_collection_count()
            logger.info(f"ChromaDB collection already contains {doc_count} documents")
            return True
            
    except Exception as e:
        logger.error(f"Error during data initialization: {e}")
        return False


def display_banner():
    """Display application banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                     TechGropse Virtual Representative                      ║
║                           Powered by CrewAI                                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  An intelligent virtual assistant to help you with all aspects of          ║
║  TechGropse - services, pricing, privacy, careers, and more.                ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def display_help():
    """Display help information."""
    help_text = """
Available commands:
  python main.py              - Start the interactive chatbot
  python main.py --health     - Check system health
  python main.py --init       - Initialize/reinitialize data
  python main.py --stats      - Show system statistics
  python main.py --help       - Show this help message

Environment variables (can be set in .env file):
  OPENAI_API_KEY              - OpenAI API key (required)
  REDIS_HOST                  - Redis host (default: localhost)
  REDIS_PORT                  - Redis port (default: 6379)
  REDIS_PASSWORD              - Redis password (if required)

Examples:
  export OPENAI_API_KEY="your-api-key-here"
  python main.py
    """
    print(help_text)


def show_system_stats():
    """Display system statistics."""
    try:
        # ChromaDB stats
        chromadb_client = ChromaDBClient()
        collection_info = chromadb_client.get_collection_info()
        
        # Session stats
        active_sessions = session_manager.get_all_sessions()
        
        print("\n📊 System Statistics")
        print("=" * 50)
        print(f"ChromaDB Collection: {collection_info.get('name', 'N/A')}")
        print(f"Document Count: {collection_info.get('document_count', 0)}")
        print(f"Embedding Model: {collection_info.get('embedding_model', 'N/A')}")
        print(f"Active Sessions: {len(active_sessions)}")
        print(f"Session Timeout: {config.session_timeout} seconds")
        print(f"Data File: {config.data_file_path}")
        
    except Exception as e:
        print(f"❌ Error getting system stats: {e}")


def main():
    """Main application entry point."""
    try:
        # Parse command line arguments
        if len(sys.argv) > 1:
            command = sys.argv[1].lower()
            
            if command == "--help":
                display_help()
                return
            elif command == "--health":
                display_banner()
                chatbot = ChatBot()
                health = chatbot.health_check()
                print("\n🏥 Health Check Results:")
                print("=" * 50)
                for component, status in health.items():
                    status_emoji = "✅" if status == "healthy" else "❌"
                    print(f"{status_emoji} {component}: {status}")
                return
            elif command == "--init":
                display_banner()
                print("🔄 Initializing data...")
                if initialize_data_if_needed():
                    print("✅ Data initialization completed")
                else:
                    print("❌ Data initialization failed")
                return
            elif command == "--stats":
                show_system_stats()
                return
            else:
                print(f"Unknown command: {command}")
                print("Use --help to see available commands")
                return
        
        # Default behavior: start interactive chatbot
        display_banner()
        
        # Check environment
        issues = check_environment()
        if issues:
            print("❌ Environment check failed:")
            for issue in issues:
                print(f"   • {issue}")
            print("\nPlease fix these issues before running the chatbot.")
            print("Use --help for more information.")
            return
        
        # Initialize data if needed
        print("🔄 Checking data initialization...")
        if not initialize_data_if_needed():
            print("❌ Data initialization failed. Cannot start chatbot.")
            return
        
        # Start chatbot
        print("🚀 Starting chatbot...")
        chatbot = ChatBot()
        chatbot.run_interactive()
        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()