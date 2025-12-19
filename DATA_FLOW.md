# Chatbot Data Flow Documentation

## Overview
This document explains the complete data flow of the TechGropse Privacy Policy Chatbot system.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            USER INPUT                                    │
│                               ↓                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                        main.py                                    │  │
│  │  - Entry point                                                    │  │
│  │  - Handles CLI arguments (--health, --init, --stats)             │  │
│  │  - Starts interactive chatbot                                     │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                               ↓                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                      chatbot.py                                   │  │
│  │  ChatBot Class:                                                   │  │
│  │  - start_session()                                                │  │
│  │  - process_message(user_input)                                    │  │
│  │  - end_session()                                                  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Data Flow

### 1. Session Initialization
```
User starts chatbot
    ↓
main.py → ChatBot.__init__()
    ↓
Initialize components:
    - ChromaDBClient (vector store)
    - SessionManager (Redis)
    - ChatbotAgent (LLM)
    ↓
ChatBot.start_session()
    ↓
SessionManager.create_session()
    ↓
Generate UUID → Store in Redis
    Key: session:{session_id}
    Value: {created_at, last_activity, query_count}
    ↓
Display welcome message
```

### 2. Processing User Query

```
User types: "what are the types of cookies?"
    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    chatbot.process_message()                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  STEP 1: Update Session Activity                                        │
│  ────────────────────────────────                                       │
│  SessionManager.update_session_activity()                               │
│    → Update last_activity timestamp in Redis                            │
│    → Increment query_count                                              │
│                                                                          │
│  STEP 2: Append to History                                              │
│  ──────────────────────────                                             │
│  SessionManager.append_message_to_history(role='user', message=input)   │
│    → Redis List: session:{session_id}:history                           │
│    → Push: {role: 'user', message: '...', timestamp: '...'}             │
│                                                                          │
│  STEP 3: Check Cache (Exact Match)                                      │
│  ──────────────────────────────────                                     │
│  SessionManager.get_cached_response(user_input)                         │
│    ↓                                                                     │
│  Normalize query: "what are the types of cookies"                       │
│    ↓                                                                     │
│  Check Redis key: cache:{session_id}:{hash(normalized_query)}           │
│    ↓                                                                     │
│  If FOUND → Return cached response ✓                                    │
│  If NOT FOUND → Continue to Step 4 ↓                                    │
│                                                                          │
│  STEP 4: Semantic Similarity Check (if cache exists)                    │
│  ────────────────────────────────────────────────────                   │
│  SessionManager._find_similar_cached_response()                         │
│    ↓                                                                     │
│  Get all cached queries from Redis                                      │
│    Keys: cache:{session_id}:*                                           │
│    ↓                                                                     │
│  If NO cached queries → Skip LLM call ✓                                 │
│  If cached queries exist → Continue ↓                                   │
│    ↓                                                                     │
│  LLM CALL #1: Semantic Similarity Check                                 │
│  ───────────────────────────────────────                                │
│  SessionManager._find_similar_query_index()                             │
│    → Compare current query against ALL cached queries                   │
│    → Single LLM call with prompt:                                       │
│        "Is '{new_query}' similar to any of these:                       │
│         1. {cached_query_1}                                              │
│         2. {cached_query_2}                                              │
│         ..."                                                             │
│    → Response: NUMBER or NONE                                            │
│    ↓                                                                     │
│  If SIMILAR → Return cached response ✓                                  │
│  If NOT SIMILAR → Continue to Step 5 ↓                                  │
│                                                                          │
│  STEP 5: Get Previous User Query (for follow-ups)                       │
│  ──────────────────────────────────────────────────                     │
│  SessionManager.get_last_user_query(skip_current=True)                  │
│    → Get session history from Redis                                     │
│    → Find most recent user message (excluding current)                  │
│    → Used for FOLLOWUP intent handling                                  │
│                                                                          │
│  STEP 6: Process Through Agent                                          │
│  ────────────────────────────                                           │
│  ChatbotAgent.process_user_input(user_input, last_user_query)          │
│    ↓                                                                     │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                                                                   │  │
│  │  LLM CALL #2: Intent Classification                               │  │
│  │  ──────────────────────────────────                               │  │
│  │  ChatbotAgent.classify_intent(user_input)                         │  │
│  │    → LLM analyzes input                                           │  │
│  │    → Returns: GREETING, QUERY, FOLLOWUP, GOODBYE, UNCLEAR        │  │
│  │                                                                   │  │
│  │  Based on Intent:                                                 │  │
│  │  ────────────────                                                 │  │
│  │                                                                   │  │
│  │  ┌─ IF GREETING ──────────────────────────────────────────────┐  │  │
│  │  │  LLM CALL #3: Generate Greeting                             │  │  │
│  │  │  ChatbotAgent.handle_greeting()                             │  │  │
│  │  │    → LLM generates warm welcome message                     │  │  │
│  │  │    → No caching                                              │  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │  │
│  │                                                                   │  │
│  │  ┌─ IF GOODBYE ────────────────────────────────────────────────┐  │  │
│  │  │  LLM CALL #3: Generate Goodbye                              │  │  │
│  │  │  ChatbotAgent.handle_goodbye()                              │  │  │
│  │  │    → LLM generates farewell message                         │  │  │
│  │  │    → No caching                                              │  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │  │
│  │                                                                   │  │
│  │  ┌─ IF QUERY ──────────────────────────────────────────────────┐  │  │
│  │  │  A. Retrieve from ChromaDB                                  │  │  │
│  │  │     ChatbotAgent.retrieve_relevant_documents(user_input)    │  │  │
│  │  │       ↓                                                      │  │  │
│  │  │     ChromaDBClient.search_similar_documents(query, n=3)     │  │  │
│  │  │       → Embedding query with SentenceTransformer            │  │  │
│  │  │       → Vector similarity search                            │  │  │
│  │  │       → Filter by distance < 1.5                            │  │  │
│  │  │       → Return top 3 relevant chunks                        │  │  │
│  │  │                                                              │  │  │
│  │  │  B. Generate Response                                        │  │  │
│  │  │     LLM CALL #3: Generate Response from Context             │  │  │
│  │  │     ChatbotAgent.generate_response_from_context()           │  │  │
│  │  │       → Format context from retrieved documents             │  │  │
│  │  │       → Build conversational prompt                         │  │  │
│  │  │       → LLM generates natural, human-like answer            │  │  │
│  │  │       → Response cached if > 50 chars                       │  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │  │
│  │                                                                   │  │
│  │  ┌─ IF FOLLOWUP ───────────────────────────────────────────────┐  │  │
│  │  │  A. Expand Query with Previous Context                      │  │  │
│  │  │     expanded_query = last_user_query + " - more details"    │  │  │
│  │  │       ↓                                                      │  │  │
│  │  │     ChromaDBClient.search_similar_documents(query, n=6)     │  │  │
│  │  │       → Retrieve MORE documents (6 instead of 3)            │  │  │
│  │  │                                                              │  │  │
│  │  │  B. Generate Detailed Response                              │  │  │
│  │  │     LLM CALL #3: Generate Followup Response                 │  │  │
│  │  │     ChatbotAgent._generate_followup_response()              │  │  │
│  │  │       → Prompt asks for MORE comprehensive answer           │  │  │
│  │  │       → LLM expands on previous topic                       │  │  │
│  │  │       → Response cached                                     │  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │  │
│  │                                                                   │  │
│  │  ┌─ IF UNCLEAR ────────────────────────────────────────────────┐  │  │
│  │  │  Return clarification message                               │  │  │
│  │  │    → No LLM call                                             │  │  │
│  │  │    → No caching                                              │  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │  │
│  │                                                                   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  STEP 7: Cache Response (if QUERY or FOLLOWUP)                          │
│  ───────────────────────────────────────────────                        │
│  If intent == 'query' AND len(response) > 50:                           │
│    SessionManager.cache_query_response(user_input, response)            │
│      → Normalize query                                                  │
│      → Store in Redis:                                                  │
│          Key: cache:{session_id}:{hash(normalized_query)}               │
│          Value: {                                                       │
│            original_query: "...",                                       │
│            normalized_query: "...",                                     │
│            response: "...",                                             │
│            timestamp: "..."                                             │
│          }                                                              │
│                                                                          │
│  STEP 8: Append Bot Response to History                                 │
│  ────────────────────────────────────────                               │
│  SessionManager.append_message_to_history(role='bot', message=response) │
│    → Redis List: session:{session_id}:history                           │
│    → Push: {role: 'bot', message: '...', timestamp: '...'}              │
│                                                                          │
│  STEP 9: Return Response to User                                        │
│  ─────────────────────────────────                                      │
│  Display: "🤖 Bot: {response}"                                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## LLM Call Count Per Query Type

### First Query (Empty Cache)
```
User Input
  ↓
LLM Call #1: Intent Classification
  ↓
No cache check (cache is empty)
  ↓
LLM Call #2: Generate Response
  ↓
TOTAL: 2 LLM calls
```

### Subsequent New Query (Cache Exists)
```
User Input
  ↓
LLM Call #1: Intent Classification
  ↓
LLM Call #2: Semantic Cache Check (against all cached queries)
  ↓
No match found
  ↓
LLM Call #3: Generate Response
  ↓
TOTAL: 3 LLM calls
```

### Cached/Similar Query
```
User Input
  ↓
LLM Call #1: Semantic Cache Check
  ↓
Match found! Return cached response
  ↓
TOTAL: 1 LLM call
```

### Follow-up Query
```
User: "need more information"
  ↓
LLM Call #1: Intent Classification → FOLLOWUP
  ↓
LLM Call #2: Semantic Cache Check
  ↓
No match
  ↓
Get last_user_query from history
  ↓
Retrieve 6 documents (instead of 3)
  ↓
LLM Call #3: Generate Detailed Response
  ↓
TOTAL: 3 LLM calls
```

---

## Redis Data Structure

```
Redis Keys:
├── session:{session_id}
│   Value: {
│     "created_at": "2025-11-17T13:28:04",
│     "last_activity": "2025-11-17T13:28:35",
│     "query_count": 5
│   }
│
├── session:{session_id}:history
│   List: [
│     {"role": "user", "message": "hi", "timestamp": "..."},
│     {"role": "bot", "message": "Hello! ...", "timestamp": "..."},
│     {"role": "user", "message": "what are cookies?", "timestamp": "..."},
│     {"role": "bot", "message": "Cookies are...", "timestamp": "..."}
│   ]
│
└── cache:{session_id}:{hash(query)}
    Value: {
      "original_query": "what are the types of cookies?",
      "normalized_query": "what are the types of cookies",
      "response": "Hey there! Great question...",
      "timestamp": "2025-11-17T13:28:35"
    }
```

---

## ChromaDB Data Structure

```
Collection: privacy_policy_docs
├── Document 1
│   ├── content: "TechGropse does not knowingly collect..."
│   ├── metadata: {source: "data/info.txt", chunk_index: 0}
│   └── embedding: [0.123, -0.456, 0.789, ...]
│
├── Document 2
│   ├── content: "We use cookies to enhance your experience..."
│   ├── metadata: {source: "data/info.txt", chunk_index: 1}
│   └── embedding: [0.321, 0.654, -0.987, ...]
│
└── ... (9 total chunks)
```

---

## Session Lifecycle

```
1. Start Session
   ↓
   Create UUID → Store in Redis
   Display welcome message

2. Query Loop
   ↓
   For each user input:
     - Update activity
     - Append to history
     - Check cache
     - Process with agent
     - Cache response
     - Append to history
     - Display response

3. End Session
   ↓
   Detect goodbye intent OR Ctrl+C
   ↓
   SessionManager.clear_session()
   ↓
   Delete Redis keys:
     - session:{session_id}
     - session:{session_id}:history
     - cache:{session_id}:*
   ↓
   Display goodbye message
   ↓
   Exit program
```

---

## Components Interaction Diagram

```
┌─────────────┐
│   User      │
└──────┬──────┘
       │
       │ User Input
       ↓
┌─────────────────────────────────────────────────────────────┐
│                         main.py                              │
└──────┬──────────────────────────────────────────────────────┘
       │
       │ Delegates to
       ↓
┌─────────────────────────────────────────────────────────────┐
│                       chatbot.py                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  ChatBot                                              │  │
│  │  - Orchestrates the conversation flow                │  │
│  │  - Manages session lifecycle                         │  │
│  └───────────────────────────────────────────────────────┘  │
└──┬──────────────┬──────────────┬──────────────┬────────────┘
   │              │              │              │
   │ Uses         │ Uses         │ Uses         │ Uses
   ↓              ↓              ↓              ↓
┌────────────┐ ┌──────────────┐ ┌────────────┐ ┌────────────┐
│ agent.py   │ │session_mgr.py│ │chromadb    │ │ config.py  │
│            │ │              │ │client.py   │ │            │
│ChatbotAgent│ │SessionManager│ │ChromaDB    │ │ Settings   │
│            │ │              │ │Client      │ │            │
│- Classify  │ │- Redis ops   │ │- Vector    │ │- Env vars  │
│  intent    │ │- Cache       │ │  search    │ │- OpenAI    │
│- Retrieve  │ │- History     │ │- Embeddings│ │  key       │
│  docs      │ │- Sessions    │ │            │ │            │
│- Generate  │ │              │ │            │ │            │
│  response  │ │              │ │            │ │            │
└─────┬──────┘ └──────┬───────┘ └─────┬──────┘ └────────────┘
      │                │               │
      │ LLM calls      │ Redis calls   │ ChromaDB calls
      ↓                ↓               ↓
┌────────────┐   ┌────────────┐  ┌────────────┐
│  OpenAI    │   │   Redis    │  │  ChromaDB  │
│  GPT-3.5   │   │  Database  │  │  Vector DB │
└────────────┘   └────────────┘  └────────────┘
```

---

## Key Optimizations

1. **Single LLM Call for Semantic Matching**
   - Instead of checking each cached query individually
   - One LLM call compares against ALL cached queries

2. **Skip Cache Check When Empty**
   - First query: No cache check (0 LLM calls)
   - Saves unnecessary API calls

3. **Exact Match Before Semantic Check**
   - Hash-based exact match first (fast)
   - Semantic similarity only if exact match fails

4. **Session-Based Caching**
   - Cache persists during session
   - Auto-cleared when session ends
   - Prevents stale data across sessions

5. **Smart Intent Classification**
   - LLM-based (not keyword-based)
   - Handles complex, nuanced inputs
   - Detects follow-ups intelligently

6. **Efficient Follow-up Handling**
   - Retrieves more documents (6 vs 3)
   - Uses previous query context
   - Generates comprehensive answers

---

## Summary

**Total Components:**
- 1 Main orchestrator (chatbot.py)
- 1 Agent (agent.py)
- 1 Session manager (session_manager.py)
- 1 Vector store client (chromadb_client.py)
- 1 Config module (config.py)

**External Services:**
- OpenAI API (GPT-3.5-turbo)
- Redis (session & cache storage)
- ChromaDB (vector embeddings & search)

**Average LLM Calls:**
- Greeting: 2 calls
- First query: 2 calls
- Cached query: 1 call
- New query: 3 calls
- Follow-up: 3 calls
- Goodbye: 3 calls
