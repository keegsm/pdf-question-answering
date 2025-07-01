#!/usr/bin/env python3
"""
Streamlit version of the PDF Question-Answering App
Much easier to deploy on Streamlit Cloud
"""

import streamlit as st
import os
import uuid
import tempfile
from pathlib import Path

import pdfplumber
import chromadb
import requests
from sentence_transformers import SentenceTransformer
import json

# Page config
st.set_page_config(
    page_title="PDF Question Answering",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'documents' not in st.session_state:
    st.session_state.documents = []

@st.cache_resource
def load_embedding_model():
    """Load the sentence transformer model"""
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def init_chroma_db():
    """Initialize ChromaDB"""
    client = chromadb.PersistentClient(path="./chroma_data")
    try:
        collection = client.get_collection("documents")
    except:
        collection = client.create_collection("documents")
    return collection

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    """Split text into overlapping chunks"""
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_text + "\n\n" + paragraph
        else:
            current_chunk += "\n\n" + paragraph if current_chunk else paragraph
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def extract_text_from_pdf(file):
    """Extract text from uploaded PDF"""
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None
    return text.strip()

def store_document(filename, text, collection, embedding_model):
    """Store document in ChromaDB"""
    doc_id = str(uuid.uuid4())
    chunks = chunk_text(text)
    
    if not chunks:
        return None
    
    embeddings = embedding_model.encode(chunks).tolist()
    
    metadatas = [
        {
            "doc_id": doc_id,
            "filename": filename,
            "chunk_index": i,
            "chunk_text": chunk[:200] + "..." if len(chunk) > 200 else chunk
        }
        for i, chunk in enumerate(chunks)
    ]
    
    chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    
    collection.add(
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
        ids=chunk_ids
    )
    
    return {"id": doc_id, "filename": filename, "chunks": len(chunks)}

def search_documents(query, collection, embedding_model, max_results=5):
    """Search for relevant document chunks"""
    query_embedding = embedding_model.encode([query]).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=max_results,
        include=["documents", "metadatas", "distances"]
    )
    
    search_results = []
    if results["documents"] and results["documents"][0]:
        for i in range(len(results["documents"][0])):
            search_results.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "similarity": 1 - results["distances"][0][i],
                "source": results["metadatas"][0][i]["filename"]
            })
    
    return search_results

def call_groq_llm(prompt, context):
    """Call Groq API"""
    api_key = st.secrets.get("GROQ_API_KEY", "")
    if not api_key:
        return None
    
    headers = {
        "Authorization": f"Bearer {api_key}",
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
        "model": "llama3-8b-8192",
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
        return None

def call_openai_llm(prompt, context):
    """Call OpenAI API"""
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        return None
    
    headers = {
        "Authorization": f"Bearer {api_key}",
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
        "model": "gpt-3.5-turbo",
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
        return None

def get_llm_response(prompt, context):
    """Get LLM response with fallback"""
    # Try Groq first (free)
    response = call_groq_llm(prompt, context)
    if response:
        return response
    
    # Try OpenAI as fallback
    response = call_openai_llm(prompt, context)
    if response:
        return response
    
    # No LLM available
    return "I apologize, but no AI language model is currently available. Please check your API configuration."

# Main app
def main():
    st.title("📄 PDF Question Answering System")
    st.markdown("Upload PDF documents and ask questions about their content using AI")
    
    # Load models
    embedding_model = load_embedding_model()
    collection = init_chroma_db()
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📤 Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload PDF documents to ask questions about"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file not in [doc.get('file') for doc in st.session_state.documents]:
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        text = extract_text_from_pdf(uploaded_file)
                        if text:
                            doc_info = store_document(uploaded_file.name, text, collection, embedding_model)
                            if doc_info:
                                doc_info['file'] = uploaded_file
                                st.session_state.documents.append(doc_info)
                                st.success(f"✅ {uploaded_file.name} processed!")
        
        # Show uploaded documents
        st.header("📚 Documents")
        if st.session_state.documents:
            for doc in st.session_state.documents:
                st.text(f"📄 {doc['filename']}")
                st.caption(f"   {doc['chunks']} chunks")
        else:
            st.info("No documents uploaded yet")
        
        # Demo questions
        st.header("💡 Demo Questions")
        demo_questions = [
            "How do I get started?",
            "What are the system requirements?",
            "How do I troubleshoot issues?",
            "What features are available?"
        ]
        
        for question in demo_questions:
            if st.button(question, key=f"demo_{question}"):
                st.session_state.messages.append({"role": "user", "content": question})
                st.rerun()
    
    # Main chat area
    st.header("💬 Ask Questions")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📖 Sources"):
                    for i, source in enumerate(message["sources"][:3]):
                        st.text(f"📄 {source['source']} (Similarity: {source['similarity']:.2f})")
                        st.caption(source['text'][:200] + "...")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            if not st.session_state.documents:
                response = "Please upload some PDF documents first before asking questions."
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                with st.spinner("Thinking..."):
                    # Search for relevant chunks
                    search_results = search_documents(prompt, collection, embedding_model)
                    
                    if search_results:
                        # Combine context
                        context = "\n\n".join([result["text"] for result in search_results[:3]])
                        
                        # Get LLM response
                        response = get_llm_response(prompt, context)
                        
                        st.markdown(response)
                        
                        # Add to message history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response,
                            "sources": search_results
                        })
                        
                        # Show sources
                        with st.expander("📖 Sources"):
                            for i, source in enumerate(search_results[:3]):
                                st.text(f"📄 {source['source']} (Similarity: {source['similarity']:.2f})")
                                st.caption(source['text'][:200] + "...")
                    else:
                        response = "I couldn't find relevant information in your documents."
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()