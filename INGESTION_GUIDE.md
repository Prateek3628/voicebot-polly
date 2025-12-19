# Enhanced Ingestion Script - Usage Guide

## Overview

The `initialise_data.py` script now supports:
1. **Multiple File Formats** - PDF, Word (.docx), TXT, Markdown (.md)
2. **Full Initialization** - Reset and reload everything
3. **Incremental Addition** - Add new files without resetting

### Supported Formats

| Format | Extension | Status | Library |
|--------|-----------|--------|---------|
| **PDF** | `.pdf` | ✅ Supported | pypdf |
| **Word** | `.docx` | ✅ Supported | python-docx |
| **Text** | `.txt` | ✅ Supported | Built-in |
| **Markdown** | `.md` | ✅ Supported | Built-in |

---

## Installation

### Install Document Processing Dependencies

```bash
# Install PDF and Word support
pip install pypdf python-docx

# Or install all requirements
pip install -r requirements.txt
```

### Verify Installation

```bash
# Test the document loader
python document_loader.py
```

**Expected output:**
```
📄 Document Loader - Supported Formats:
--------------------------------------------------
PDF (.pdf): ✅ Available
Word (.docx): ✅ Available
Text (.txt): ✅ Available
Markdown (.md): ✅ Available
```

---

## Usage Examples

### Working with Different File Formats

#### PDF Documents
```bash
# Add a PDF privacy policy
python initialise_data.py --incremental data/privacy_policy.pdf --category privacy

# Add multiple PDFs
python initialise_data.py --incremental \
    data/terms.pdf \
    data/gdpr.pdf \
    --category legal
```

#### Word Documents (.docx)
```bash
# Add a Word document
python initialise_data.py --incremental data/employee_handbook.docx --category hr

# Mix formats
python initialise_data.py --incremental \
    data/policy.docx \
    data/guidelines.pdf \
    data/faq.txt
```

#### Text Files
```bash
# Add text files (original format)
python initialise_data.py --incremental data/info.txt --category privacy
```

#### Markdown Files
```bash
# Add markdown documentation
python initialise_data.py --incremental \
    data/README.md \
    data/CONTRIBUTING.md \
    --category documentation
```

---

## Usage Modes

### Mode 1: Full Initialization (First Time)

```bash
# Initialize with default file (data/info.txt)
python initialise_data.py

# Initialize with custom file
python initialise_data.py --data-file data/privacy_policy.txt

# Force reset and reinitialize
python initialise_data.py --reset
```

**What it does:**
- Creates ChromaDB collection
- Processes and embeds documents
- Verifies embeddings with test queries

---

### Mode 2: Incremental Addition (Add New Files) ✨

```bash
# Add a single new file
python initialise_data.py --incremental data/terms_of_service.txt

# Add multiple files at once
python initialise_data.py --incremental data/cookie_policy.txt data/gdpr_compliance.txt

# Add files with custom category
python initialise_data.py --incremental data/cookie_policy.txt --category cookies
```

**What it does:**
- ✅ Keeps existing documents
- ✅ Adds new documents to collection
- ✅ No re-embedding of old data
- ✅ Enhanced metadata with category

---

## Command-Line Arguments

| Argument | Type | Description | Example |
|----------|------|-------------|---------|
| `--data-file` | String | Path to data file (init mode) | `--data-file data/policy.txt` |
| `--reset` | Flag | Reset collection before init | `--reset` |
| `--incremental` | List | Add files incrementally | `--incremental file1.txt file2.txt` |
| `--category` | String | Category for incremental docs | `--category privacy` |

---

## Workflow Examples

### Scenario 1: Initial Setup

```bash
# Step 1: First time initialization
python initialise_data.py --data-file data/privacy_policy.txt

# Output:
# ✓ ChromaDB has been initialized with privacy policy data
# ✓ Collection 'privacy_policy_docs' now contains 9 documents
```

### Scenario 2: Adding New Documents

```bash
# Step 2: Add terms of service (keeps existing privacy policy)
python initialise_data.py --incremental data/terms_of_service.txt --category terms

# Output:
# Current collection contains 9 documents
# Processing file: data/terms_of_service.txt
# ✓ Added 7 chunks from data/terms_of_service.txt
# Collection updated: 9 → 16 documents
# ✓ Successfully added 7 new document chunks
```

### Scenario 3: Adding Multiple Files

```bash
# Step 3: Add multiple policy documents
python initialise_data.py --incremental \
    data/cookie_policy.txt \
    data/data_retention.txt \
    data/gdpr_compliance.txt \
    --category compliance

# Output:
# Current collection contains 16 documents
# Processing file: data/cookie_policy.txt
# ✓ Added 5 chunks from data/cookie_policy.txt
# Processing file: data/data_retention.txt
# ✓ Added 4 chunks from data/data_retention.txt
# Processing file: data/gdpr_compliance.txt
# ✓ Added 6 chunks from data/gdpr_compliance.txt
# Collection updated: 16 → 31 documents
# ✓ Successfully added 15 new document chunks
```

### Scenario 4: Full Reset (Re-ingest Everything)

```bash
# If you need to start fresh (e.g., changed embedding model)
python initialise_data.py --reset --data-file data/privacy_policy.txt

# Output:
# Resetting ChromaDB collection...
# Deleted existing collection: privacy_policy_docs
# Created new collection: privacy_policy_docs
# ✓ ChromaDB initialization completed successfully!
```

---

## Enhanced Metadata

Each document now includes rich metadata:

```python
{
    "source": "data/cookie_policy.txt",
    "document_type": "policy",
    "category": "cookies",           # Customizable via --category
    "company": "TechGropse",
    "version": "1.0",
    "language": "english",
    "last_updated": "2025-11-27T17:56:00",  # Auto-generated
    "domain": "privacy_and_data_protection"
}
```

**Benefits:**
- Filter searches by category
- Track document sources
- Version management
- Timestamp tracking

---

## Filtering by Metadata (Future Use)

```python
# Example: Search only cookie-related documents
from vectorstore.chromadb_client import ChromaDBClient

client = ChromaDBClient()
results = client.collection.query(
    query_texts=["cookie retention"],
    where={"category": "cookies"},  # Filter by category
    n_results=3
)
```

---

## Best Practices

### ✅ Do:
1. Use `--incremental` when adding new files
2. Specify meaningful `--category` values
3. Keep original files organized in `data/` directory
4. Test with `python main.py` after adding documents

### ❌ Don't:
1. Use `--reset` unless necessary (loses all data)
2. Mix different document types without categories
3. Add duplicate files (will create duplicate embeddings)
4. Change embedding model without `--reset`

---

## Troubleshooting

### Issue: "Collection already contains X documents"

**Solution:** This is normal! Use `--incremental` to add more:
```bash
python initialise_data.py --incremental data/new_file.txt
```

### Issue: "Data file not found"

**Solution:** Check file path is correct:
```bash
ls -la data/  # Verify file exists
python initialise_data.py --incremental data/correct_filename.txt
```

### Issue: Need to change embedding model

**Solution:** Must reset and re-ingest all data:
```bash
# 1. Update config.py with new embedding_model
# 2. Reset and reinitialize
python initialise_data.py --reset --data-file data/privacy_policy.txt
# 3. Re-add other files
python initialise_data.py --incremental data/terms.txt data/cookies.txt
```

---

## Summary

| Mode | Command | Use When |
|------|---------|----------|
| **Init** | `python initialise_data.py` | First time setup |
| **Reset** | `python initialise_data.py --reset` | Changed embedding model |
| **Add Files** | `python initialise_data.py --incremental file1.txt file2.txt` | Adding new documents |
| **With Category** | `python initialise_data.py --incremental file.txt --category privacy` | Organized metadata |

**Key Takeaway:** Use `--incremental` to add new files without re-ingesting everything! 🚀
