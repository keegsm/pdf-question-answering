#!/usr/bin/env python3
"""
Visage PACS Support Hub - Dynamic Version
PDF Question-Answering with Visage PACS branding and dynamic document processing
Based on the IO architecture with Visage-specific enhancements
"""

import streamlit as st
import os
import json
import tempfile
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
import requests

# Page configuration
st.set_page_config(
    page_title="Visage PACS Support Hub",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state.documents = {}
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'tfidf_matrix' not in st.session_state:
    st.session_state.tfidf_matrix = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = 'groq'

# Model configuration
MODELS = {
    'groq': {
        'name': 'Groq Llama 3.1 8B',
        'description': 'Fast and free (700 requests/day)',
        'speed': '⚡⚡⚡⚡',
        'quality': '⭐⭐⭐',
        'cost': '🆓',
        'api_key_env': 'GROQ_API_KEY',
        'endpoint': 'https://api.groq.com/openai/v1/chat/completions',
        'model_id': 'llama3-8b-8192'
    },
    'openai': {
        'name': 'OpenAI GPT-3.5',
        'description': 'Reliable and well-tested',
        'speed': '⚡⚡⚡',
        'quality': '⭐⭐⭐⭐',
        'cost': '💰',
        'api_key_env': 'OPENAI_API_KEY',
        'endpoint': 'https://api.openai.com/v1/chat/completions',
        'model_id': 'gpt-3.5-turbo'
    },
    'openrouter': {
        'name': 'OpenRouter',
        'description': 'Access to multiple models',
        'speed': '⚡⚡⚡',
        'quality': '⭐⭐⭐⭐',
        'cost': '💰',
        'api_key_env': 'OPENROUTER_API_KEY',
        'endpoint': 'https://openrouter.ai/api/v1/chat/completions',
        'model_id': 'meta-llama/llama-3.1-8b-instruct:free'
    }
}

# Visage demo questions
VISAGE_DEMO_QUESTIONS = [
    "What are the system requirements for Visage 7 PACS?",
    "How do I set up user permissions in Visage PACS?",
    "How do I configure DICOM settings in Visage PACS?",
    "How do I search for studies in Visage PACS?",
    "What should I do if images are not loading in PACS?",
    "How do I troubleshoot PACS login issues?",
    "How do I export studies from Visage PACS?",
    "How do I configure study routing in Visage PACS?",
    "How do I manage PACS storage and archiving?",
    "How do I use the Visage PACS web viewer?"
]

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from uploaded PDF file"""
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""
    return text.strip()

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    if not text:
        return []
    
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.strip()) > 50:  # Only include substantial chunks
            chunks.append(chunk.strip())
    
    return chunks

def build_search_index():
    """Build TF-IDF search index from all document chunks"""
    if not st.session_state.chunks:
        return
    
    try:
        st.session_state.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        st.session_state.tfidf_matrix = st.session_state.vectorizer.fit_transform(st.session_state.chunks)
    except Exception as e:
        st.error(f"Error building search index: {str(e)}")

def search_documents(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search for relevant document chunks"""
    if not st.session_state.vectorizer or not st.session_state.tfidf_matrix:
        return []
    
    try:
        query_vector = st.session_state.vectorizer.transform([query.lower()])
        similarities = cosine_similarity(query_vector, st.session_state.tfidf_matrix).flatten()
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.01:  # Minimum similarity threshold
                results.append({
                    'text': st.session_state.chunks[idx],
                    'similarity': float(similarities[idx]),
                    'chunk_index': idx
                })
        
        return results
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []

def call_llm_api(prompt: str, context: str, model_key: str) -> str:
    """Call LLM API with the selected model"""
    model = MODELS[model_key]
    api_key = os.getenv(model['api_key_env'])
    
    if not api_key:
        return f"❌ API key not configured for {model['name']}. Please add {model['api_key_env']} to your Streamlit secrets."
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    if model_key == 'openrouter':
        headers["HTTP-Referer"] = "https://streamlit.io"
        headers["X-Title"] = "Visage PACS Support Hub"
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant specializing in Visage PACS support. Answer questions based on the provided context from Visage documentation. If the context doesn't contain enough information, say so clearly."
        },
        {
            "role": "user",
            "content": f"Context from Visage documentation:\n{context}\n\nQuestion: {prompt}\n\nPlease provide a clear, accurate answer based on the Visage documentation provided."
        }
    ]
    
    payload = {
        "model": model['model_id'],
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 500
    }
    
    try:
        response = requests.post(model['endpoint'], headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ Error calling {model['name']}: {str(e)}"

def process_pdf_file(uploaded_file, progress_bar, status_text):
    """Process uploaded PDF file"""
    filename = uploaded_file.name
    
    # Step 1: Extract text (50% progress)
    status_text.text(f"📄 Processing {filename}: Extracting text...")
    progress_bar.progress(25)
    
    text = extract_text_from_pdf(uploaded_file)
    if not text:
        st.error(f"❌ Could not extract text from {filename}")
        return False
    
    progress_bar.progress(50)
    
    # Step 2: Chunk text (25% progress)
    status_text.text(f"📄 Processing {filename}: Creating text chunks...")
    chunks = chunk_text(text)
    progress_bar.progress(75)
    
    # Step 3: Store document (25% progress)
    status_text.text(f"📄 Processing {filename}: Storing document...")
    
    doc_info = {
        'filename': filename,
        'text_length': len(text),
        'chunk_count': len(chunks),
        'uploaded_at': datetime.now().isoformat()
    }
    
    st.session_state.documents[filename] = doc_info
    st.session_state.chunks.extend(chunks)
    
    progress_bar.progress(100)
    status_text.text(f"✅ {filename} processed successfully!")
    
    return True

def main():
    # Custom CSS for Visage branding
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79, #2e6da4);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 {
        margin: 0;
        color: white !important;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    .search-result {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #337ab7;
    }
    .model-card {
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin: 0.5rem 0;
    }
    .model-card.selected {
        border-color: #337ab7;
        background-color: #f0f8ff;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📡 Visage PACS Support Hub</h1>
        <p>Upload Visage documentation and get instant AI-powered answers</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - Model Selection and Configuration
    with st.sidebar:
        st.markdown("### 🤖 AI Model Selection")
        
        for model_key, model_info in MODELS.items():
            api_key_configured = bool(os.getenv(model_info['api_key_env']))
            status_icon = "✅" if api_key_configured else "❌"
            
            if st.button(
                f"{status_icon} {model_info['name']}", 
                key=f"select_{model_key}",
                use_container_width=True,
                type="primary" if st.session_state.selected_model == model_key else "secondary"
            ):
                st.session_state.selected_model = model_key
            
            if st.session_state.selected_model == model_key:
                st.markdown(f"""
                <div class="model-card selected">
                    <strong>{model_info['name']}</strong><br>
                    {model_info['description']}<br>
                    Speed: {model_info['speed']}<br>
                    Quality: {model_info['quality']}<br>
                    Cost: {model_info['cost']}
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📊 Document Status")
        st.write(f"**Uploaded:** {len(st.session_state.documents)} documents")
        st.write(f"**Text chunks:** {len(st.session_state.chunks)}")
        
        if st.session_state.documents:
            st.markdown("### 📋 Uploaded Documents")
            for filename, doc_info in st.session_state.documents.items():
                st.write(f"• **{filename}**")
                st.write(f"  └ {doc_info['chunk_count']} chunks")
        
        # API Configuration Help
        st.markdown("---")
        st.markdown("### 🔑 API Configuration")
        st.info("""
        Add API keys to Streamlit secrets:
        - `GROQ_API_KEY` (free, 700 req/day)
        - `OPENAI_API_KEY` (paid)
        - `OPENROUTER_API_KEY` (paid)
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📤 Upload Visage Documents")
        
        uploaded_files = st.file_uploader(
            "Upload Visage PACS/RIS PDFs",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload your Visage documentation PDFs"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.documents:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    if process_pdf_file(uploaded_file, progress_bar, status_text):
                        # Rebuild search index after adding new document
                        build_search_index()
                        st.rerun()
        
        # Demo questions
        st.markdown("### 💡 Example Questions")
        for i, question in enumerate(VISAGE_DEMO_QUESTIONS[:5]):  # Show first 5
            if st.button(question, key=f"demo_{i}", use_container_width=True):
                st.session_state.current_question = question
    
    with col2:
        st.markdown("### 💬 Ask Questions")
        
        # Chat input
        question = st.text_input(
            "Ask about your Visage documentation:",
            value=getattr(st.session_state, 'current_question', ''),
            placeholder="e.g., How do I configure DICOM settings?",
            key="question_input"
        )
        
        if st.button("🔍 Search & Answer", type="primary", use_container_width=True):
            if not question:
                st.warning("Please enter a question.")
            elif not st.session_state.chunks:
                st.warning("Please upload some Visage documents first.")
            else:
                with st.spinner("🔎 Searching documentation..."):
                    # Search for relevant chunks
                    search_results = search_documents(question, top_k=3)
                    
                    if search_results:
                        # Create context from top results
                        context = "\n\n".join([result['text'] for result in search_results])
                        
                        # Get AI answer
                        with st.spinner("🤖 Generating answer..."):
                            answer = call_llm_api(question, context, st.session_state.selected_model)
                        
                        # Add to chat history
                        st.session_state.chat_messages.append({
                            'question': question,
                            'answer': answer,
                            'sources': search_results,
                            'timestamp': datetime.now().isoformat(),
                            'model': MODELS[st.session_state.selected_model]['name']
                        })
                        
                        # Clear current question
                        if hasattr(st.session_state, 'current_question'):
                            del st.session_state.current_question
                        
                        st.rerun()
                    else:
                        st.warning("No relevant information found in your documents. Try rephrasing your question.")
        
        # Display chat history
        if st.session_state.chat_messages:
            st.markdown("### 📚 Previous Questions & Answers")
            
            for i, chat in enumerate(reversed(st.session_state.chat_messages[-5:])):  # Show last 5
                with st.expander(f"Q: {chat['question'][:60]}...", expanded=(i == 0)):
                    st.markdown(f"**🤖 Answer ({chat['model']}):**")
                    st.markdown(chat['answer'])
                    
                    if chat['sources']:
                        st.markdown("**📖 Sources:**")
                        for j, source in enumerate(chat['sources'], 1):
                            similarity_percent = int(source['similarity'] * 100)
                            st.markdown(f"""
                            <div class="search-result">
                                <strong>Source {j}</strong> ({similarity_percent}% relevance)
                                <p>{source['text'][:300]}...</p>
                            </div>
                            """, unsafe_allow_html=True)
        
        # Clear chat history button
        if st.session_state.chat_messages:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_messages = []
                st.rerun()

if __name__ == "__main__":
    main()
