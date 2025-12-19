# Privacy Policy Chatbot

An intelligent chatbot powered by CrewAI that helps users understand TechGropse's privacy policy and answers questions about data collection, usage, and user rights.

## Features

- **CrewAI Agent**: Single intelligent agent for intent classification, greeting handling, and query processing
- **Redis Session Management**: Session-based caching with automatic cleanup
- **ChromaDB Integration**: Vector database for efficient document retrieval
- **Smart Caching**: Cached responses with contextual prefixes ("As I mentioned earlier...")
- **Intent Classification**: Automatic detection of greetings, queries, and goodbye messages
- **Socket.IO Integration**: Real-time communication with web interface
- **Interactive Modes**: Both command-line and web-based interfaces

## Architecture Flow

```
User Query → Agent (Intent Classification) → Cache Check → Response Generation
                ↓                              ↓
        Greeting/Query/Goodbye         Cache Hit → Return with prefix
                ↓                              ↓
        If Query → ChromaDB Retrieval   Cache Miss → Process & Cache
```

## Installation

1. **Clone and navigate to the project directory**
   ```bash
   cd simple_chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start Redis server** (required for session management)
   ```bash
   # macOS with Homebrew
   brew install redis
   brew services start redis
   
   # Ubuntu/Debian
   sudo apt update && sudo apt install redis-server
   sudo systemctl start redis-server
   
   # Or use Docker
   docker run -d -p 6379:6379 redis:alpine
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

5. **Initialize data** (optional - will auto-initialize on first run)
   ```bash
   python main.py --init
   ```

## Usage

### Web Interface (Socket.IO)
```bash
# Start the web server with real-time chat
python run_socket_server.py
# OR
python socket_server.py

# Open browser to: http://localhost:5000
```

### Command Line Interface
```bash
python main.py
```

### Available Commands
```bash
python main.py --help        # Show help information
python main.py --health      # Check system health
python main.py --init        # Initialize/reinitialize data
python main.py --stats       # Show system statistics
```

### Socket Events

**Client → Server:**
- `user_query`: Send a message to the bot
- `health_check`: Request system health status
- `get_stats`: Request session statistics

**Server → Client:**
- `query_received`: Acknowledgment that query was received
- `bot_response`: Bot's response to the query
- `status`: System status messages
- `error`: Error messages

### Example Conversation
```
🤖 Privacy Policy Assistant

Hello! I'm here to help you understand TechGropse's privacy policy...

💬 You: Hello
🤖 Bot: Hello! I'm here to help you with any questions about our privacy policy. What would you like to know?

💬 You: What data do you collect?
🤖 Bot: We collect several types of personal information when you use our services:
- Your name, email address, and phone number when you contact us
- Company name, address, and telephone number when you register
- Message content and attachments you send to us
...

💬 You: What data do you collect?
🤖 Bot: As I mentioned earlier, we collect several types of personal information when you use our services...

💬 You: bye
🤖 Bot: Thank you for your questions! If you need any more information about our privacy policy, feel free to ask anytime.
```

## Configuration

### Environment Variables
Set these in your `.env` file:

- `OPENAI_API_KEY` (Required): Your OpenAI API key
- `REDIS_HOST` (Optional): Redis host (default: localhost)
- `REDIS_PORT` (Optional): Redis port (default: 6379)
- `REDIS_PASSWORD` (Optional): Redis password if required

### System Configuration
Edit `config.py` to modify:
- Chunking parameters
- Session timeout
- Embedding model
- Data file path

## Components

### 1. CrewAI Agent (`agent.py`)
- Single agent for all tasks
- Intent classification (greeting, query, goodbye)
- Context-aware response generation
- ChromaDB integration for document retrieval

### 2. Session Manager (`session_manager.py`)
- Redis-based session management
- Query-response caching
- Automatic session cleanup
- Activity tracking

### 3. ChromaDB Client (`vectorstore/chromadb_client.py`)
- Document chunking and embedding
- Vector similarity search
- Persistent storage

### 4. Main Interface (`chatbot.py`)
- Orchestrates the conversation flow
- Cache-first strategy
- Session management
- Error handling

## Data Initialization

The system automatically processes the privacy policy document (`data/info.txt`) into chunks and stores them in ChromaDB with embeddings. This enables semantic search for relevant information.

To manually reinitialize data:
```bash
python main.py --init
```

## Session Management

- Each interaction creates a unique session
- Sessions automatically expire after 1 hour of inactivity
- Query-response pairs are cached per session
- Cache is cleared when session ends
- Cached responses include contextual prefixes

## Health Monitoring

Check system health:
```bash
python main.py --health
```

This verifies:
- Redis connection
- ChromaDB collection status
- Agent functionality
- Overall system health

## Troubleshooting

### Common Issues

1. **Redis Connection Error**
   ```bash
   # Ensure Redis is running
   redis-cli ping  # Should return "PONG"
   ```

2. **OpenAI API Key Missing**
   ```bash
   # Set in .env file
   OPENAI_API_KEY=your-api-key-here
   ```

3. **Empty ChromaDB Collection**
   ```bash
   # Reinitialize data
   python main.py --init
   ```

### Logs
The application logs important events and errors. Check the console output for debugging information.

## Dependencies

- **crewai**: AI agent framework
- **langchain**: LLM integration and text processing
- **chromadb**: Vector database
- **redis**: Session and cache management
- **sentence-transformers**: Text embeddings

## License

This project is for educational and internal use.