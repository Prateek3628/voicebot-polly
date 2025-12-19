# File Usage Analysis

## 🟢 ACTIVELY USED FILES

### Core Application Files
| File | Purpose | Used By |
|------|---------|---------|
| `main.py` | CLI chatbot interface | Direct execution |
| `chatbot.py` | Main chatbot class | main.py, socket_server.py, text_to_voice_server.py |
| `agent.py` | AI agent logic & LLM integration | chatbot.py |
| `config.py` | Configuration management | All modules |
| `session_manager.py` | Redis session management | chatbot.py |

### Server Files
| File | Purpose | Status |
|------|---------|--------|
| `socket_server.py` | Socket.IO server for web clients | ✅ Active |
| `text_to_voice_server.py` | Socket.IO server with TTS | ✅ Active |
| `run_socket_server.py` | Server launcher script | ✅ Active |

### Data & Storage
| File | Purpose | Used By |
|------|---------|---------|
| `document_loader.py` | Load PDF/DOCX/TXT files | initialise_data.py |
| `initialise_data.py` | Initialize vector database | main.py, manual execution |
| `vectorstore/chromadb_client.py` | ChromaDB vector store | agent.py, initialise_data.py |

### Contact Form System
| File | Purpose | Used By |
|------|---------|---------|
| `contact_form_handler.py` | Multi-step contact form | chatbot.py |
| `database/mongodb_client.py` | MongoDB operations | contact_form_handler.py |
| `utils/validators.py` | Input validation | contact_form_handler.py |

---

## 🟡 UTILITY/SUPPORT FILES

### Extension Files
| File | Purpose | Status |
|------|---------|--------|
| `session_manager_extensions.py` | Session manager helpers | ⚠️ May be unused (check imports) |

---

## 🔴 UNUSED/REDUNDANT FILES

### Backup Files
| File | Reason | Action |
|------|--------|--------|
| `socket_server.py.backup` | Backup from fix | ❌ Can delete |

### Test Files (Keep for testing)
| File | Purpose | Keep? |
|------|---------|-------|
| `test_concurrent_sessions.py` | Concurrent session tests | ✅ Keep |
| `test_socketio_concurrent.py` | Socket.IO tests | ✅ Keep |
| `test_mongodb.py` | MongoDB tests | ✅ Keep |
| `test_socket_client.py` | Socket client test | ⚠️ Check if needed |

### Documentation Files (Keep)
| File | Purpose |
|------|---------|
| `README.md` | Project documentation |
| `DATA_FLOW.md` | Data flow documentation |
| `INGESTION_GUIDE.md` | Data ingestion guide |
| `MULTI_DOCUMENT_FIX.md` | Multi-document fix notes |
| `MULTI_FORMAT_GUIDE.md` | Multi-format support guide |
| `DYNAMIC_UNCLEAR_HANDLING.md` | Unclear query handling |
| `TESTING_CONCURRENT.md` | Concurrent testing guide |
| `architecture_design.md.resolved` | Architecture design |

---

## 📊 Summary

### Definitely UNUSED (Safe to Delete)
```
socket_server.py.backup  # Backup file from concurrent fix
```

### Potentially UNUSED (Need to Check)
```
session_manager_extensions.py  # Check if imported anywhere
test_socket_client.py          # Check if still needed
```

### KEEP (Active Use)
```
# Core
main.py
chatbot.py
agent.py
config.py
session_manager.py

# Servers
socket_server.py
text_to_voice_server.py
run_socket_server.py

# Data
document_loader.py
initialise_data.py
vectorstore/chromadb_client.py

# Contact Form
contact_form_handler.py
database/mongodb_client.py
utils/validators.py

# Tests (for verification)
test_concurrent_sessions.py
test_socketio_concurrent.py
test_mongodb.py

# Documentation
All .md files
```

---

## 🔍 Detailed Check: session_manager_extensions.py

Let me check if this file is actually imported anywhere...
