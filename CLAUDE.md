# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PDF Question-Answering web application built with FastAPI that allows users to upload PDF documents and ask questions about their content using AI. The system prioritizes cost-effectiveness with free local LLM options while maintaining professional demo capabilities.

## Development Commands

### Basic Operations
```bash
# Install dependencies
pip install -r requirements.txt

# Run application (development)
python main.py

# Quick demo startup (cross-platform)
./run_demo.sh        # Linux/Mac
run_demo.bat          # Windows
```

### LLM Backend Setup
```bash
# Install Ollama (free local LLM)
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull phi3:mini      # 2.3GB model for 4GB RAM
ollama pull llama3.1:8b    # 4.7GB model for 8GB RAM

# Check Ollama status
curl http://localhost:11434/api/tags
```

### Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Required for cloud LLMs (optional)
export GROQ_API_KEY=your_groq_key_here        # Free 700 req/day
export OPENAI_API_KEY=your_openai_key_here    # Paid service
```

## Architecture

### Core Application Structure

**Single-File Backend (`main.py`):**
- FastAPI application with all endpoints and business logic
- PDF processing pipeline: upload → text extraction → chunking → vector storage
- Multi-LLM backend system with automatic fallback prioritization
- ChromaDB vector database with sentence-transformers embeddings

**Frontend Architecture:**
- Single-page application (`templates/index.html`) with vanilla JavaScript
- Class-based JavaScript architecture (`PDFQuestionApp`) in `static/script.js`
- Professional CSS with CSS variables and responsive design in `static/style.css`

### LLM Backend System

The application uses a **hierarchical LLM backend system** with automatic fallback:

1. **Ollama** (local, free) - checked via `check_ollama_availability()`
2. **Groq** (cloud, free tier) - 700 requests/day
3. **OpenAI** (cloud, paid) - fallback for production

**Key function:** `get_llm_response(prompt, context)` tries backends in priority order and automatically fails over.

Each LLM backend has its own `call_*_llm()` function with specific API formatting and error handling.

### Document Processing Pipeline

1. **Upload** (`/upload` endpoint): PDF → file storage with UUID naming
2. **Text Extraction**: `extract_text_from_pdf()` using pdfplumber
3. **Chunking**: `chunk_text()` with paragraph-aware splitting (500 chars, 50 char overlap)
4. **Vector Storage**: `store_document_chunks()` generates embeddings and stores in ChromaDB
5. **Search**: `search_documents()` performs semantic similarity search

### Vector Database Architecture

- **ChromaDB** persistent client with local storage in `vector_db/`
- **Embeddings**: sentence-transformers `all-MiniLM-L6-v2` model (local, free)
- **Document Storage**: Each PDF split into chunks with metadata (doc_id, filename, chunk_index)
- **Search Strategy**: Semantic similarity with distance-to-similarity conversion

## Key API Endpoints

- `GET /` - Main web interface
- `POST /upload` - PDF upload and processing
- `POST /ask` - Question answering with vector search
- `GET /docs` - List uploaded documents
- `DELETE /docs/{doc_id}` - Remove documents and chunks
- `GET /demo` - Demo questions from `demo_data/demo_questions.json`
- `GET /health` - System status including LLM backend availability

## Frontend JavaScript Architecture

**Main Class:** `PDFQuestionApp` singleton pattern with:
- **State management**: documents, chat history, system health
- **Event handling**: file upload, drag/drop, chat input, demo questions
- **API integration**: all backend communication via fetch()
- **UI management**: toast notifications, loading overlays, modal dialogs

**Key methods:**
- `handleFileUpload()` - processes multiple PDFs with progress tracking
- `sendMessage()` - chat interface with typing indicators and source attribution
- `checkSystemHealth()` - periodic health checks with status indicator updates

## Environment Variables

**LLM Configuration:**
- `GROQ_API_KEY` - Groq API key (free tier available)
- `OPENAI_API_KEY` - OpenAI API key (paid service)
- `OLLAMA_HOST` - Ollama server URL (default: http://localhost:11434)

**Application Settings:**
- `PORT` - Server port (default: 8000)
- `DEBUG` - Development mode flag
- `MAX_FILE_SIZE_MB` - Upload limit (default: 10MB)
- `VECTOR_DB_PATH` - ChromaDB storage path

## Demo and Deployment Features

### Demo Mode
- Pre-configured demo questions in `demo_data/demo_questions.json`
- Sample documentation in `demo_data/sample_user_guide.txt`
- One-click startup scripts for cross-platform demos
- Real-time system status indicators showing LLM backend availability

### Deployment Architecture
- **Portable**: Self-contained with local dependencies
- **Demo-ready**: Professional UI suitable for client presentations
- **Offline capable**: Works completely offline with Ollama
- **Cost-optimized**: Prioritizes free options with paid fallbacks

## Development Patterns

### Adding New LLM Backends
1. Add configuration to `LLM_BACKENDS` dictionary
2. Implement `call_newllm_llm(prompt, context)` function
3. Add backend name to priority list in `get_llm_response()`
4. Update environment variables and documentation

### PDF Processing Extensions
- Text extraction handled by `extract_text_from_pdf()` using pdfplumber
- Chunking strategy in `chunk_text()` preserves paragraph boundaries
- Vector storage via `store_document_chunks()` with metadata preservation

### Frontend State Management
- Centralized state in `PDFQuestionApp.state` object
- DOM element references cached in `this.elements`
- Event handlers use arrow functions to preserve `this` context
- API calls use async/await with try/catch error handling

## Configuration Management

The application uses environment variables with fallbacks to defaults. Critical settings:
- LLM API keys are optional (system works without them using Ollama)
- File upload limits and vector database paths are configurable
- CORS is enabled for demo purposes (consider restricting in production)

The system is designed to work out-of-the-box with minimal configuration while supporting advanced customization through environment variables.