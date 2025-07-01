#!/usr/bin/env python3
"""
InteleOrchestrator Assistant - Domain-Specific Q&A App
Pre-configured for InteleOrchestrator system support with auto-loaded knowledge base
"""

import streamlit as st
import os
import json
import uuid
import time
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
import requests
import re
import glob

# Load app configuration
@st.cache_data
def load_app_config():
    """Load app configuration from JSON file"""
    try:
        with open('app_config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("App configuration file not found. Please ensure app_config.json exists.")
        st.stop()
    except json.JSONDecodeError:
        st.error("Invalid app configuration file. Please check app_config.json format.")
        st.stop()

# Load configuration
config = load_app_config()

# Page config
st.set_page_config(
    page_title=config["app_info"]["title"],
    page_icon=config["app_info"]["icon"],
    layout="wide",
    initial_sidebar_state="collapsed"  # Simplified UI - no sidebar by default
)

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'messages': [],
        'documents_processed': False,
        'document_chunks': [],
        'vectorizer': None,
        'tfidf_matrix': None,
        'processing_progress': {},
        'knowledge_base_loaded': False,
        'usage_stats': {'requests_today': 0, 'total_requests': 0}
    }
    
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

def get_api_key() -> Optional[str]:
    """Get API key for the configured provider"""
    provider = config["model_config"]["provider"]
    if provider == "Groq":
        return st.secrets.get("GROQ_API_KEY", "")
    elif provider == "OpenAI":
        return st.secrets.get("OPENAI_API_KEY", "")
    elif provider == "OpenRouter":
        return st.secrets.get("OPENROUTER_API_KEY", "")
    return None

def is_model_available() -> bool:
    """Check if the configured model is available"""
    return bool(get_api_key())

def chunk_text(text: str) -> List[str]:
    """Split text into overlapping chunks"""
    chunk_size = config["knowledge_base"]["chunk_size"]
    overlap = config["knowledge_base"]["chunk_overlap"]
    
    # Clean text
    text = re.sub(r'\s+', ' ', text).strip()
    
    sentences = text.split('.')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        if len(current_chunk) + len(sentence) + 1 > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Start new chunk with overlap
            words = current_chunk.split()
            overlap_words = words[-overlap//10:] if len(words) > overlap//10 else words
            current_chunk = ' '.join(overlap_words) + '. ' + sentence
        else:
            current_chunk = current_chunk + '. ' + sentence if current_chunk else sentence
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return [chunk for chunk in chunks if len(chunk.strip()) > 50]

def extract_text_from_pdf(file_path: str, progress_callback=None) -> Optional[str]:
    """Extract text from PDF file with progress tracking"""
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            
            for page_num, page in enumerate(pdf.pages):
                # Update progress
                if progress_callback:
                    progress = (page_num + 1) / total_pages
                    filename = os.path.basename(file_path)
                    progress_callback(f"Processing {filename} - page {page_num + 1} of {total_pages}", progress)
                
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
                
                # Small delay to show progress
                time.sleep(0.05)
                
    except Exception as e:
        st.error(f"Error extracting text from {os.path.basename(file_path)}: {str(e)}")
        return None
    
    return text.strip()

def load_knowledge_base():
    """Auto-load all documents from knowledge base directory"""
    if st.session_state.knowledge_base_loaded:
        return True
    
    knowledge_base_dir = config["knowledge_base"]["documents_folder"]
    if not os.path.exists(knowledge_base_dir):
        st.error(f"Knowledge base directory '{knowledge_base_dir}' not found.")
        return False
    
    # Find all supported document files
    supported_formats = config["knowledge_base"]["supported_formats"]
    document_files = []
    for fmt in supported_formats:
        pattern = os.path.join(knowledge_base_dir, f"*.{fmt}")
        document_files.extend(glob.glob(pattern))
    
    if not document_files:
        st.error(f"No documents found in '{knowledge_base_dir}' directory.")
        return False
    
    # Initialize progress tracking
    st.session_state.processing_progress = {
        'current_operation': 'Initializing knowledge base...',
        'overall_progress': 0.0,
        'start_time': time.time(),
        'total_documents': len(document_files)
    }
    
    # Progress container
    progress_container = st.empty()
    
    with progress_container.container():
        st.info("🔄 Loading InteleOrchestrator knowledge base...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_chunks = []
        
        for doc_index, file_path in enumerate(document_files):
            filename = os.path.basename(file_path)
            
            # Update overall progress
            def progress_callback(message, file_progress):
                overall_progress = (doc_index + file_progress) / len(document_files)
                st.session_state.processing_progress.update({
                    'current_operation': message,
                    'overall_progress': overall_progress
                })
                progress_bar.progress(overall_progress)
                status_text.text(message)
            
            # Extract text based on file type
            if file_path.endswith('.pdf'):
                text = extract_text_from_pdf(file_path, progress_callback)
            elif file_path.endswith(('.txt', '.md')):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    progress_callback(f"Processing {filename}", 1.0)
                except Exception as e:
                    st.error(f"Error reading {filename}: {str(e)}")
                    continue
            else:
                continue
            
            if text and len(text.strip()) > 100:
                # Chunk the text
                chunks = chunk_text(text)
                
                # Add chunks to search index
                for i, chunk in enumerate(chunks):
                    chunk_info = {
                        'id': f"{filename}_chunk_{i}",
                        'filename': filename,
                        'chunk_index': i,
                        'text': chunk
                    }
                    all_chunks.append(chunk_info)
        
        # Build search index
        if all_chunks:
            status_text.text("Building search index...")
            progress_bar.progress(0.9)
            
            st.session_state.document_chunks = all_chunks
            
            # Create TF-IDF vectorizer
            chunk_texts = [chunk['text'] for chunk in all_chunks]
            st.session_state.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                lowercase=True
            )
            
            # Fit and transform
            st.session_state.tfidf_matrix = st.session_state.vectorizer.fit_transform(chunk_texts)
            
            progress_bar.progress(1.0)
            status_text.text(f"✅ Successfully loaded {len(document_files)} documents with {len(all_chunks)} text chunks")
            
            time.sleep(1)  # Show completion briefly
            st.session_state.knowledge_base_loaded = True
            st.session_state.processing_progress = {}
            
            # Clear progress display
            progress_container.empty()
            return True
        else:
            st.error("No meaningful text found in any documents.")
            return False

def search_documents(query: str, max_results: int = 5) -> List[Dict]:
    """Search for relevant document chunks using TF-IDF"""
    if not st.session_state.document_chunks or st.session_state.vectorizer is None:
        return []
    
    # Transform query
    query_vector = st.session_state.vectorizer.transform([query])
    
    # Calculate similarities
    similarities = cosine_similarity(query_vector, st.session_state.tfidf_matrix).flatten()
    
    # Get top results
    top_indices = similarities.argsort()[-max_results:][::-1]
    
    results = []
    for idx in top_indices:
        if similarities[idx] > 0.1:  # Minimum similarity threshold
            chunk = st.session_state.document_chunks[idx]
            results.append({
                'text': chunk['text'],
                'source': chunk['filename'],
                'similarity': float(similarities[idx]),
                'chunk_index': chunk['chunk_index']
            })
    
    return results

def get_llm_response(prompt: str, context: str) -> Optional[Dict]:
    """Get LLM response using configured model"""
    api_key = get_api_key()
    if not api_key:
        return {
            "content": f"API key for {config['model_config']['provider']} is not configured. Please add it in Streamlit secrets.",
            "model": config["model_config"]["model_name"],
            "provider": config["model_config"]["provider"],
            "response_time": 0,
            "tokens_used": "N/A",
            "error": True
        }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Add provider-specific headers
    if config["model_config"]["provider"] == "OpenRouter":
        headers["HTTP-Referer"] = "https://inteleorchestrator-assistant.streamlit.app"
        headers["X-Title"] = "InteleOrchestrator Assistant"
    
    messages = [
        {
            "role": "system",
            "content": config["system_prompt"]
        },
        {
            "role": "user",
            "content": f"Context from InteleOrchestrator documentation:\n\n{context}\n\nQuestion: {prompt}\n\nPlease provide a clear, helpful answer based on the context above."
        }
    ]
    
    payload = {
        "model": config["model_config"]["model_id"],
        "messages": messages,
        "temperature": config["model_config"]["temperature"],
        "max_tokens": config["model_config"]["max_tokens"]
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            config["model_config"]["api_url"],
            headers=headers,
            json=payload,
            timeout=30
        )
        response_time = time.time() - start_time
        
        response.raise_for_status()
        result = response.json()
        
        return {
            "content": result["choices"][0]["message"]["content"],
            "model": config["model_config"]["model_name"],
            "provider": config["model_config"]["provider"],
            "response_time": round(response_time, 2),
            "tokens_used": result.get("usage", {}).get("total_tokens", "Unknown")
        }
        
    except Exception as e:
        st.error(f"{config['model_config']['provider']} API error: {str(e)}")
        return None

def main():
    # Initialize session state
    init_session_state()
    
    # App header
    st.title(f"{config['app_info']['icon']} {config['app_info']['title']}")
    st.markdown(config["app_info"]["description"])
    
    # Role-based quick access
    with st.expander("🎯 Quick Start by Role", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**👩‍⚕️ Medical Staff**")
            for question in config.get("role_based_examples", {}).get("medical", []):
                if st.button(question, key=f"medical_{question}", use_container_width=True):
                    st.session_state.quick_question = question
                    st.rerun()
        
        with col2:
            st.markdown("**👨‍💼 Administrators**")
            for question in config.get("role_based_examples", {}).get("admin", []):
                if st.button(question, key=f"admin_{question}", use_container_width=True):
                    st.session_state.quick_question = question
                    st.rerun()
        
        with col3:
            st.markdown("**🔧 IT Support**")
            for question in config.get("role_based_examples", {}).get("it", []):
                if st.button(question, key=f"it_{question}", use_container_width=True):
                    st.session_state.quick_question = question
                    st.rerun()
    
    # Auto-load knowledge base if not already loaded
    if not st.session_state.knowledge_base_loaded:
        if not load_knowledge_base():
            st.stop()
    
    # Check model availability
    if not is_model_available():
        st.error(f"⚠️ {config['model_config']['provider']} API key is not configured. Please add it in Streamlit secrets to use the assistant.")
        st.info(f"Add `{config['model_config']['provider'].upper()}_API_KEY` to your Streamlit secrets.")
        st.stop()
    
    # Main chat interface
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.header("💬 InteleOrchestrator Support")
    
    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show response metadata for assistant messages
                if message["role"] == "assistant" and "metadata" in message:
                    metadata = message["metadata"]
                    if not metadata.get("error", False):
                        cols = st.columns(4)
                        with cols[0]:
                            st.caption(f"🤖 {metadata['model']}")
                        with cols[1]:
                            st.caption(f"⏱️ {metadata['response_time']}s")
                        with cols[2]:
                            st.caption(f"🎯 {metadata['provider']}")
                        with cols[3]:
                            st.caption(f"📊 {metadata['tokens_used']} tokens")
                
                # Show sources
                if "sources" in message and message["sources"]:
                    with st.expander("📖 Sources", expanded=False):
                        for i, source in enumerate(message["sources"][:3]):
                            st.markdown(f"**📄 {source['source']}** (Relevance: {source['similarity']:.1%})")
                            st.caption(source['text'][:300] + ("..." if len(source['text']) > 300 else ""))
                            if i < len(message["sources"]) - 1:
                                st.divider()
    
    # Handle quick questions
    if hasattr(st.session_state, 'quick_question'):
        prompt = st.session_state.quick_question
        delattr(st.session_state, 'quick_question')
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate response
        with st.spinner("Searching InteleOrchestrator documentation and generating answer..."):
            # Search for relevant chunks
            search_results = search_documents(prompt, max_results=5)
            
            if search_results:
                # Combine context from top results
                context = "\n\n".join([f"From {result['source']}:\n{result['text']}" 
                                     for result in search_results[:3]])
                
                # Get LLM response
                response_data = get_llm_response(prompt, context)
                
                if response_data:
                    # Add to message history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response_data["content"],
                        "sources": search_results,
                        "metadata": response_data
                    })
                    
                    # Update usage stats
                    st.session_state.usage_stats['requests_today'] += 1
                    st.session_state.usage_stats['total_requests'] += 1
            else:
                response = "I couldn't find relevant information in the InteleOrchestrator documentation to answer that question. Please try rephrasing your question or contact technical support."
                st.session_state.messages.append({"role": "assistant", "content": response})
        
        st.rerun()
    
    # Chat input
    if prompt := st.chat_input("Ask about workflows, administration, troubleshooting, or training..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate response
        with st.spinner("Searching InteleOrchestrator documentation and generating answer..."):
            # Search for relevant chunks
            search_results = search_documents(prompt, max_results=5)
            
            if search_results:
                # Combine context from top results
                context = "\n\n".join([f"From {result['source']}:\n{result['text']}" 
                                     for result in search_results[:3]])
                
                # Get LLM response
                response_data = get_llm_response(prompt, context)
                
                if response_data:
                    # Add to message history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response_data["content"],
                        "sources": search_results,
                        "metadata": response_data
                    })
                    
                    # Update usage stats
                    st.session_state.usage_stats['requests_today'] += 1
                    st.session_state.usage_stats['total_requests'] += 1
            else:
                response = "I couldn't find relevant information in the InteleOrchestrator documentation to answer that question. Please try rephrasing your question or contact technical support."
                st.session_state.messages.append({"role": "assistant", "content": response})
        
        st.rerun()

    # Footer info
    st.markdown("---")
    
    # Support information
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("**📋 Support Escalation:**")
        st.markdown("• **Complex technical issues** → Contact IT Support or system vendor")
        st.markdown("• **Clinical workflow questions** → Refer to department training coordinator")
        st.markdown("• **Urgent system problems** → Follow your organization's emergency IT procedures")
    
    with col2:
        with st.expander("📊 System Stats"):
            st.metric("Documents", len(glob.glob(os.path.join(config["knowledge_base"]["documents_folder"], "*"))))
            st.metric("Text Chunks", len(st.session_state.document_chunks))
            st.metric("Questions Asked", len([m for m in st.session_state.messages if m["role"] == "user"]))
            status = "🟢 Ready" if is_model_available() else "🔴 API Issue"
            st.caption(f"**Status:** {status}")

if __name__ == "__main__":
    main()