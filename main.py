#!/usr/bin/env python3
"""
PDF Question-Answering Web App
A FastAPI application that allows users to upload PDFs, extract text, and ask questions
using vector search with local or cloud LLMs.
"""

import os
import json
import uuid
import asyncio
from typing import List, Dict, Optional, Any
from pathlib import Path

import pdfplumber
import chromadb
import requests
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="PDF Question Answering", description="Upload PDFs and ask questions using AI", version="1.0.0")

# Enable CORS for demo purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize directories
UPLOAD_DIR = Path("uploads")
VECTOR_DB_DIR = Path("vector_db")
DEMO_DATA_DIR = Path("demo_data")
UPLOAD_DIR.mkdir(exist_ok=True)
VECTOR_DB_DIR.mkdir(exist_ok=True)

# Initialize ChromaDB and embedding model
chroma_client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Get or create collection
try:
    collection = chroma_client.get_collection("documents")
except:
    collection = chroma_client.create_collection("documents")

# Pydantic models
class QuestionRequest(BaseModel):
    question: str
    max_results: int = 5

class DocumentInfo(BaseModel):
    id: str
    filename: str
    upload_time: str
    chunk_count: int

class AnswerResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float

# LLM Configuration
LLM_BACKENDS = {
    "ollama": {
        "url": "http://localhost:11434/api/generate",
        "model": "llama3.1:8b",
        "available": False
    },
    "groq": {
        "api_key": os.getenv("GROQ_API_KEY"),
        "model": "llama3-8b-8192",
        "available": bool(os.getenv("GROQ_API_KEY"))
    },
    "openai": {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "model": "gpt-3.5-turbo",
        "available": bool(os.getenv("OPENAI_API_KEY"))
    }
}


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks while preserving paragraph boundaries."""
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If adding this paragraph exceeds chunk size, save current chunk
        if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Start new chunk with overlap from previous chunk
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_text + "\n\n" + paragraph
        else:
            current_chunk += "\n\n" + paragraph if current_chunk else paragraph
    
    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF using pdfplumber."""
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error extracting text from PDF: {str(e)}")
    
    return text.strip()


def store_document_chunks(doc_id: str, filename: str, chunks: List[str]) -> None:
    """Store document chunks in ChromaDB with embeddings."""
    embeddings = embedding_model.encode(chunks).tolist()
    
    # Create metadata for each chunk
    metadatas = [
        {
            "doc_id": doc_id,
            "filename": filename,
            "chunk_index": i,
            "chunk_text": chunk[:200] + "..." if len(chunk) > 200 else chunk
        }
        for i, chunk in enumerate(chunks)
    ]
    
    # Create unique IDs for each chunk
    chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    
    # Add to collection
    collection.add(
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
        ids=chunk_ids
    )


def search_documents(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search for relevant document chunks using semantic similarity."""
    query_embedding = embedding_model.encode([query]).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=max_results,
        include=["documents", "metadatas", "distances"]
    )
    
    search_results = []
    for i in range(len(results["documents"][0])):
        search_results.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "similarity": 1 - results["distances"][0][i],  # Convert distance to similarity
            "source": results["metadatas"][0][i]["filename"]
        })
    
    return search_results


def check_ollama_availability() -> bool:
    """Check if Ollama is available and running."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False


def call_ollama_llm(prompt: str, context: str) -> str:
    """Call local Ollama LLM."""
    full_prompt = f"""Context information:
{context}

Question: {prompt}

Please provide a clear, accurate answer based only on the context provided. If the context doesn't contain enough information to answer the question, say "I don't have enough information in the provided documents to answer this question."

Answer:"""

    payload = {
        "model": LLM_BACKENDS["ollama"]["model"],
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 500
        }
    }
    
    try:
        response = requests.post(LLM_BACKENDS["ollama"]["url"], json=payload, timeout=30)
        response.raise_for_status()
        return response.json().get("response", "No response generated.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama API error: {str(e)}")


def call_groq_llm(prompt: str, context: str) -> str:
    """Call Groq API."""
    headers = {
        "Authorization": f"Bearer {LLM_BACKENDS['groq']['api_key']}",
        "Content-Type": "application/json"
    }
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that answers questions based on provided context. If the context doesn't contain enough information, say so clearly."
        },
        {
            "role": "user",
            "content": f"Context: {context}\n\nQuestion: {prompt}\n\nPlease provide a clear answer based only on the context provided."
        }
    ]
    
    payload = {
        "model": LLM_BACKENDS["groq"]["model"],
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 500
    }
    
    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", 
                               headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API error: {str(e)}")


def call_openai_llm(prompt: str, context: str) -> str:
    """Call OpenAI API."""
    headers = {
        "Authorization": f"Bearer {LLM_BACKENDS['openai']['api_key']}",
        "Content-Type": "application/json"
    }
    
    messages = [
        {
            "role": "system", 
            "content": "You are a helpful assistant that answers questions based on provided context. If the context doesn't contain enough information, say so clearly."
        },
        {
            "role": "user",
            "content": f"Context: {context}\n\nQuestion: {prompt}\n\nPlease provide a clear answer based only on the context provided."
        }
    ]
    
    payload = {
        "model": LLM_BACKENDS["openai"]["model"],
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 500
    }
    
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions",
                               headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")


def get_llm_response(prompt: str, context: str) -> str:
    """Get LLM response using available backends in priority order."""
    # Update Ollama availability
    LLM_BACKENDS["ollama"]["available"] = check_ollama_availability()
    
    # Try backends in order of preference
    for backend_name in ["ollama", "groq", "openai"]:
        backend = LLM_BACKENDS[backend_name]
        if backend["available"]:
            try:
                if backend_name == "ollama":
                    return call_ollama_llm(prompt, context)
                elif backend_name == "groq":
                    return call_groq_llm(prompt, context)
                elif backend_name == "openai":
                    return call_openai_llm(prompt, context)
            except Exception as e:
                print(f"Error with {backend_name}: {e}")
                continue
    
    # Fallback response if no LLM is available
    return "I apologize, but no AI language model is currently available to answer your question. Please check your API configuration or install Ollama for local processing."


# Routes

@app.get("/")
async def home(request: Request):
    """Main page with upload and chat interface."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF file."""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Generate unique document ID
    doc_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{doc_id}_{file.filename}"
    
    try:
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Extract text from PDF
        text = extract_text_from_pdf(str(file_path))
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")
        
        # Chunk the text
        chunks = chunk_text(text)
        
        # Store in vector database
        store_document_chunks(doc_id, file.filename, chunks)
        
        return JSONResponse({
            "message": "PDF uploaded and processed successfully",
            "doc_id": doc_id,
            "filename": file.filename,
            "chunks": len(chunks),
            "text_length": len(text)
        })
        
    except Exception as e:
        # Clean up file if processing failed
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """Answer a question using uploaded documents."""
    try:
        # Search for relevant document chunks
        search_results = search_documents(request.question, request.max_results)
        
        if not search_results:
            return AnswerResponse(
                answer="I don't have any documents to search through. Please upload some PDF documents first.",
                sources=[],
                confidence=0.0
            )
        
        # Combine context from top results
        context = "\n\n".join([result["text"] for result in search_results[:3]])
        
        # Get LLM response
        answer = get_llm_response(request.question, context)
        
        # Calculate confidence based on similarity scores
        avg_similarity = sum(result["similarity"] for result in search_results[:3]) / min(3, len(search_results))
        
        return AnswerResponse(
            answer=answer,
            sources=search_results,
            confidence=avg_similarity
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@app.get("/documents")
async def list_documents():
    """List all uploaded documents."""
    try:
        # Get all unique documents from the collection
        all_results = collection.get(include=["metadatas"])
        
        docs = {}
        for metadata in all_results["metadatas"]:
            doc_id = metadata["doc_id"]
            if doc_id not in docs:
                docs[doc_id] = {
                    "id": doc_id,
                    "filename": metadata["filename"],
                    "chunk_count": 0
                }
            docs[doc_id]["chunk_count"] += 1
        
        return list(docs.values())
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document and all its chunks."""
    try:
        # Get all chunk IDs for this document
        results = collection.get(
            where={"doc_id": doc_id},
            include=["metadatas"]
        )
        
        if not results["ids"]:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete from vector database
        collection.delete(ids=results["ids"])
        
        # Delete physical file if it exists
        for file_path in UPLOAD_DIR.glob(f"{doc_id}_*"):
            file_path.unlink()
        
        return {"message": "Document deleted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")


@app.get("/demo")
async def get_demo_questions():
    """Get demo questions for testing."""
    try:
        demo_file = DEMO_DATA_DIR / "demo_questions.json"
        if demo_file.exists():
            with open(demo_file, 'r') as f:
                return json.load(f)
        return {"demo_questions": [], "sample_responses": {}}
    except Exception as e:
        return {"demo_questions": [], "sample_responses": {}, "error": str(e)}


@app.get("/health")
async def health_check():
    """System health check for demos."""
    # Check LLM availability
    LLM_BACKENDS["ollama"]["available"] = check_ollama_availability()
    
    return {
        "status": "healthy",
        "embedding_model": "all-MiniLM-L6-v2",
        "vector_db": "ChromaDB",
        "llm_backends": {
            name: {"available": config["available"], "model": config.get("model", "N/A")}
            for name, config in LLM_BACKENDS.items()
        },
        "documents_count": len(collection.get()["ids"]),
        "upload_dir": str(UPLOAD_DIR),
        "demo_mode": DEMO_DATA_DIR.exists()
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print("Starting PDF Question-Answering Web App...")
    print(f"Server will run on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)