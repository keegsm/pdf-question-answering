#!/usr/bin/env python3
"""
Enhanced Streamlit PDF Question-Answering App
Features: Multiple LLM providers, progress tracking, model selection UI
"""

import streamlit as st
import os
import uuid
import time
import asyncio
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
import requests
import json
import re

# Page config
st.set_page_config(
    page_title="PDF Question Answering Pro",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# LLM Model Configuration
@dataclass
class LLMModel:
    name: str
    provider: str
    model_id: str
    api_url: str
    free: bool
    speed: str  # "Fast", "Medium", "Slow"
    quality: str  # "Good", "Better", "Best"
    description: str

# Available LLM Models
AVAILABLE_MODELS = {
    "groq_llama3_8b": LLMModel(
        name="Llama 3 8B",
        provider="Groq",
        model_id="llama3-8b-8192",
        api_url="https://api.groq.com/openai/v1/chat/completions",
        free=True,
        speed="Fast",
        quality="Good",
        description="Fast, reliable model from Meta"
    ),
    "groq_llama3_70b": LLMModel(
        name="Llama 3 70B",
        provider="Groq",
        model_id="llama3-70b-8192",
        api_url="https://api.groq.com/openai/v1/chat/completions",
        free=True,
        speed="Medium",
        quality="Better",
        description="Larger, more capable Llama model"
    ),
    "groq_mixtral": LLMModel(
        name="Mixtral 8x7B",
        provider="Groq", 
        model_id="mixtral-8x7b-32768",
        api_url="https://api.groq.com/openai/v1/chat/completions",
        free=True,
        speed="Medium",
        quality="Better",
        description="Mixture of experts model by Mistral"
    ),
    "openrouter_deepseek": LLMModel(
        name="DeepSeek V3",
        provider="OpenRouter",
        model_id="deepseek/deepseek-chat",
        api_url="https://openrouter.ai/api/v1/chat/completions",
        free=True,
        speed="Medium",
        quality="Best",
        description="Latest DeepSeek model - very capable"
    ),
    "openrouter_llama405b": LLMModel(
        name="Llama 3.1 405B",
        provider="OpenRouter",
        model_id="meta-llama/llama-3.1-405b-instruct:free",
        api_url="https://openrouter.ai/api/v1/chat/completions",
        free=True,
        speed="Slow",
        quality="Best",
        description="Massive 405B parameter model"
    ),
    "openrouter_qwen": LLMModel(
        name="Qwen 2.5 72B",
        provider="OpenRouter",
        model_id="qwen/qwen-2.5-72b-instruct:free",
        api_url="https://openrouter.ai/api/v1/chat/completions",
        free=True,
        speed="Medium",
        quality="Better",
        description="Alibaba's powerful language model"
    ),
    "openai_gpt35": LLMModel(
        name="GPT-3.5 Turbo",
        provider="OpenAI",
        model_id="gpt-3.5-turbo",
        api_url="https://api.openai.com/v1/chat/completions",
        free=False,
        speed="Fast",
        quality="Better",
        description="OpenAI's efficient model"
    ),
    "openai_gpt4omini": LLMModel(
        name="GPT-4o Mini",
        provider="OpenAI",
        model_id="gpt-4o-mini",
        api_url="https://api.openai.com/v1/chat/completions",
        free=False,
        speed="Medium",
        quality="Best",
        description="OpenAI's latest efficient model"
    )
}

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'messages': [],
        'documents': [],
        'document_chunks': [],
        'vectorizer': None,
        'tfidf_matrix': None,
        'selected_model': 'groq_llama3_8b',
        'processing_progress': {},
        'api_status': {},
        'usage_stats': {'requests_today': 0, 'total_requests': 0},
        'processing_queue': []
    }
    
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

# LLM Provider Classes
class LLMProvider:
    """Base class for LLM providers"""
    
    def __init__(self, model: LLMModel):
        self.model = model
    
    def get_api_key(self) -> Optional[str]:
        """Get API key from secrets"""
        if self.model.provider == "Groq":
            return st.secrets.get("GROQ_API_KEY", "")
        elif self.model.provider == "OpenAI":
            return st.secrets.get("OPENAI_API_KEY", "")
        elif self.model.provider == "OpenRouter":
            return st.secrets.get("OPENROUTER_API_KEY", "")
        return None
    
    def is_available(self) -> bool:
        """Check if provider is available"""
        return bool(self.get_api_key())
    
    def call_api(self, messages: List[Dict], **kwargs) -> Optional[Dict]:
        """Make API call to LLM provider"""
        api_key = self.get_api_key()
        if not api_key:
            return None
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Add OpenRouter specific headers
        if self.model.provider == "OpenRouter":
            headers["HTTP-Referer"] = "https://pdf-qa-app.streamlit.app"
            headers["X-Title"] = "PDF Question Answering"
        
        payload = {
            "model": self.model.model_id,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.1),
            "max_tokens": kwargs.get("max_tokens", 500)
        }
        
        try:
            start_time = time.time()
            response = requests.post(
                self.model.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response_time = time.time() - start_time
            
            response.raise_for_status()
            result = response.json()
            
            return {
                "content": result["choices"][0]["message"]["content"],
                "model": self.model.name,
                "provider": self.model.provider,
                "response_time": round(response_time, 2),
                "tokens_used": result.get("usage", {}).get("total_tokens", "Unknown")
            }
            
        except Exception as e:
            st.error(f"{self.model.provider} API error: {str(e)}")
            return None

def get_available_models() -> List[str]:
    """Get list of available models based on API keys"""
    available = []
    
    for model_key, model in AVAILABLE_MODELS.items():
        provider = LLMProvider(model)
        if provider.is_available():
            available.append(model_key)
    
    # Add fallback if no APIs available
    if not available:
        available = ["groq_llama3_8b"]  # Will show error when used
    
    return available

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks with progress tracking"""
    # Clean text
    text = re.sub(r'\s+', ' ', text).strip()
    
    sentences = text.split('.')
    chunks = []
    current_chunk = ""
    
    total_sentences = len(sentences)
    
    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Update progress
        progress = (i + 1) / total_sentences
        if 'chunking_progress' in st.session_state.processing_progress:
            st.session_state.processing_progress['chunking_progress'] = progress
        
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

def extract_text_from_pdf(file, progress_callback=None) -> Optional[str]:
    """Extract text from uploaded PDF with progress tracking"""
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            total_pages = len(pdf.pages)
            
            for page_num, page in enumerate(pdf.pages):
                # Update progress
                if progress_callback:
                    progress = (page_num + 1) / total_pages
                    progress_callback(f"Extracting text from page {page_num + 1} of {total_pages}", progress)
                
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
                
                # Small delay to show progress
                time.sleep(0.1)
                
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None
    
    return text.strip()

def update_search_index():
    """Update the TF-IDF search index with progress tracking"""
    if not st.session_state.document_chunks:
        return
    
    # Update progress
    st.session_state.processing_progress['indexing_progress'] = 0.5
    st.session_state.processing_progress['current_operation'] = "Building search index..."
    
    # Get all chunk texts
    chunk_texts = [chunk['text'] for chunk in st.session_state.document_chunks]
    
    # Create TF-IDF vectorizer
    st.session_state.vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2),
        lowercase=True
    )
    
    # Fit and transform
    st.session_state.tfidf_matrix = st.session_state.vectorizer.fit_transform(chunk_texts)
    
    # Complete indexing
    st.session_state.processing_progress['indexing_progress'] = 1.0

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
    """Get LLM response using selected model"""
    selected_model_key = st.session_state.selected_model
    model = AVAILABLE_MODELS[selected_model_key]
    provider = LLMProvider(model)
    
    if not provider.is_available():
        return {
            "content": f"API key for {model.provider} is not configured. Please add it in Streamlit secrets.",
            "model": model.name,
            "provider": model.provider,
            "response_time": 0,
            "tokens_used": "N/A",
            "error": True
        }
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that answers questions based on provided context. Be concise and accurate. If the context doesn't contain enough information, say so clearly."
        },
        {
            "role": "user",
            "content": f"Context from documents:\n\n{context}\n\nQuestion: {prompt}\n\nPlease provide a clear, helpful answer based on the context above."
        }
    ]
    
    return provider.call_api(messages)

def render_model_selector():
    """Render the LLM model selection interface"""
    st.sidebar.header("🤖 AI Model Selection")
    
    available_models = get_available_models()
    
    # Model selection
    model_options = {}
    for model_key in available_models:
        model = AVAILABLE_MODELS[model_key]
        status_icon = "🆓" if model.free else "💰"
        speed_icon = {"Fast": "⚡", "Medium": "⚖️", "Slow": "🐌"}[model.speed]
        quality_icon = {"Good": "⭐", "Better": "⭐⭐", "Best": "⭐⭐⭐"}[model.quality]
        
        display_name = f"{status_icon} {model.name} ({model.provider}) {speed_icon}{quality_icon}"
        model_options[display_name] = model_key
    
    if model_options:
        current_display = None
        for display, key in model_options.items():
            if key == st.session_state.selected_model:
                current_display = display
                break
        
        selected_display = st.sidebar.selectbox(
            "Choose AI Model:",
            options=list(model_options.keys()),
            index=list(model_options.keys()).index(current_display) if current_display else 0,
            help="🆓 Free • 💰 Paid • ⚡ Fast • ⚖️ Medium • 🐌 Slow • ⭐ Quality Rating"
        )
        
        st.session_state.selected_model = model_options[selected_display]
    
    # Show current model info
    current_model = AVAILABLE_MODELS[st.session_state.selected_model]
    provider = LLMProvider(current_model)
    
    status_color = "green" if provider.is_available() else "red"
    status_text = "Ready" if provider.is_available() else "API Key Missing"
    
    st.sidebar.markdown(f"**Status:** :{status_color}[{status_text}]")
    st.sidebar.markdown(f"**Description:** {current_model.description}")
    
    # Usage stats
    if st.session_state.usage_stats['total_requests'] > 0:
        st.sidebar.markdown(f"**Today's Usage:** {st.session_state.usage_stats['requests_today']} requests")

def render_processing_progress():
    """Render PDF processing progress"""
    if st.session_state.processing_progress:
        st.sidebar.header("⏳ Processing Status")
        
        current_op = st.session_state.processing_progress.get('current_operation', 'Processing...')
        st.sidebar.text(current_op)
        
        # Overall progress
        if 'overall_progress' in st.session_state.processing_progress:
            progress = st.session_state.processing_progress['overall_progress']
            st.sidebar.progress(progress)
            st.sidebar.caption(f"{int(progress * 100)}% complete")
        
        # Time estimates
        if 'eta_seconds' in st.session_state.processing_progress:
            eta = st.session_state.processing_progress['eta_seconds']
            if eta > 0:
                st.sidebar.caption(f"ETA: {int(eta)} seconds")

def process_pdf_with_progress(uploaded_file) -> bool:
    """Process PDF with detailed progress tracking"""
    try:
        # Initialize progress tracking
        st.session_state.processing_progress = {
            'current_operation': f'Starting to process {uploaded_file.name}...',
            'overall_progress': 0.0,
            'start_time': time.time()
        }
        
        # Stage 1: Text Extraction (50% of total)
        def extraction_callback(message, progress):
            st.session_state.processing_progress.update({
                'current_operation': message,
                'overall_progress': progress * 0.5,
                'extraction_progress': progress
            })
        
        text = extract_text_from_pdf(uploaded_file, extraction_callback)
        if not text or len(text.strip()) < 100:
            st.error(f"No meaningful text found in {uploaded_file.name}")
            return False
        
        # Stage 2: Text Chunking (25% of total)
        st.session_state.processing_progress.update({
            'current_operation': 'Splitting text into chunks...',
            'overall_progress': 0.5,
            'chunking_progress': 0.0
        })
        
        chunks = chunk_text(text)
        if not chunks:
            st.error(f"Could not create text chunks from {uploaded_file.name}")
            return False
        
        st.session_state.processing_progress['overall_progress'] = 0.75
        
        # Stage 3: Document Storage (25% of total)
        st.session_state.processing_progress.update({
            'current_operation': 'Storing document and building index...',
            'overall_progress': 0.75
        })
        
        # Add document info
        doc_info = {
            'id': str(uuid.uuid4()),
            'filename': uploaded_file.name,
            'chunks': len(chunks),
            'text_length': len(text),
            'processed_at': datetime.now().strftime("%H:%M:%S")
        }
        st.session_state.documents.append(doc_info)
        
        # Add chunks to search index
        for i, chunk in enumerate(chunks):
            chunk_info = {
                'id': f"{doc_info['id']}_chunk_{i}",
                'doc_id': doc_info['id'],
                'filename': uploaded_file.name,
                'chunk_index': i,
                'text': chunk
            }
            st.session_state.document_chunks.append(chunk_info)
        
        # Update search index
        update_search_index()
        
        # Complete processing
        total_time = time.time() - st.session_state.processing_progress['start_time']
        st.session_state.processing_progress.update({
            'current_operation': f'✅ {uploaded_file.name} processed successfully!',
            'overall_progress': 1.0,
            'completed': True,
            'total_time': round(total_time, 1)
        })
        
        return True
        
    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        st.session_state.processing_progress = {}
        return False

def main():
    # Initialize session state
    init_session_state()
    
    st.title("📄 PDF Question Answering Pro")
    st.markdown("Upload PDF documents and ask questions using multiple AI models")
    
    # Sidebar
    with st.sidebar:
        # Model selection
        render_model_selector()
        
        # Processing progress
        render_processing_progress()
        
        st.header("📤 Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload PDF documents to ask questions about"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Check if file already processed
                if not any(doc['filename'] == uploaded_file.name for doc in st.session_state.documents):
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        success = process_pdf_with_progress(uploaded_file)
                        if success:
                            st.success(f"✅ {uploaded_file.name} processed!")
                            # Clear progress after delay
                            time.sleep(2)
                            st.session_state.processing_progress = {}
                            st.rerun()
        
        # Show uploaded documents
        st.header("📚 Documents")
        if st.session_state.documents:
            for doc in st.session_state.documents:
                with st.expander(f"📄 {doc['filename']}", expanded=False):
                    st.write(f"**Chunks:** {doc['chunks']}")
                    st.write(f"**Size:** {doc['text_length']:,} characters")
                    st.write(f"**Processed:** {doc['processed_at']}")
                    
                    if st.button(f"🗑️ Delete", key=f"delete_{doc['id']}"):
                        # Remove document and its chunks
                        st.session_state.documents = [d for d in st.session_state.documents if d['id'] != doc['id']]
                        st.session_state.document_chunks = [c for c in st.session_state.document_chunks if c['doc_id'] != doc['id']]
                        
                        # Update search index
                        if st.session_state.document_chunks:
                            update_search_index()
                        else:
                            st.session_state.vectorizer = None
                            st.session_state.tfidf_matrix = None
                        
                        st.success(f"Deleted {doc['filename']}")
                        st.rerun()
        else:
            st.info("No documents uploaded yet")
        
        # Clear all button
        if st.session_state.documents:
            if st.button("🗑️ Clear All Documents", type="secondary"):
                st.session_state.documents = []
                st.session_state.document_chunks = []
                st.session_state.vectorizer = None
                st.session_state.tfidf_matrix = None
                st.success("All documents cleared!")
                st.rerun()
        
        # Demo questions
        st.header("💡 Demo Questions")
        demo_questions = [
            "How do I get started?",
            "What are the system requirements?", 
            "How do I troubleshoot issues?",
            "What features are available?",
            "Where can I find installation instructions?"
        ]
        
        for question in demo_questions:
            if st.button(question, key=f"demo_{question}", use_container_width=True):
                st.session_state.demo_question = question
                st.rerun()
    
    # Main chat area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Show current model in header
        current_model = AVAILABLE_MODELS[st.session_state.selected_model]
        st.header(f"💬 Chat - Using {current_model.name}")
    
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
    
    # Handle demo question
    if hasattr(st.session_state, 'demo_question'):
        prompt = st.session_state.demo_question
        delattr(st.session_state, 'demo_question')
        
        # Process demo question
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        if not st.session_state.documents:
            response = "Please upload some PDF documents first before asking questions."
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            with st.spinner("Searching documents and generating answer..."):
                search_results = search_documents(prompt, max_results=5)
                
                if search_results:
                    context = "\n\n".join([f"From {result['source']}:\n{result['text']}" 
                                         for result in search_results[:3]])
                    
                    response_data = get_llm_response(prompt, context)
                    if response_data:
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
                    response = "I couldn't find relevant information in your uploaded documents to answer that question."
                    st.session_state.messages.append({"role": "assistant", "content": response})
        
        st.rerun()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate response
        if not st.session_state.documents:
            response = "Please upload some PDF documents first before asking questions."
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            with st.spinner("Searching documents and generating answer..."):
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
                    response = "I couldn't find relevant information in your uploaded documents to answer that question."
                    st.session_state.messages.append({"role": "assistant", "content": response})
        
        st.rerun()

    # Footer info
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Documents", len(st.session_state.documents))
    with col2:
        st.metric("Text Chunks", len(st.session_state.document_chunks))
    with col3:
        st.metric("Chat Messages", len(st.session_state.messages))
    with col4:
        current_model = AVAILABLE_MODELS[st.session_state.selected_model]
        status = "🟢" if LLMProvider(current_model).is_available() else "🔴"
        st.metric("AI Status", status)

if __name__ == "__main__":
    main()