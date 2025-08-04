#!/usr/bin/env python3
"""
Visage PACS Support Hub - Optimized Version
Ultra-fast loading with pre-processed Visage PACS knowledge base
"""

import streamlit as st
import os
import json
import pickle
import time
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import requests

# Load app configuration
@st.cache_data
def load_app_config():
    """Load app configuration from JSON file"""
    # Return Visage PACS configuration
    return {
        "app_info": {
            "title": "Visage PACS Support Hub",
            "icon": "📡",
            "description": "**AI-powered support for Visage PACS documentation and troubleshooting**\n\n*Get instant answers from official Visage PACS manuals, setup guides, and troubleshooting documentation.*"
        },
        "model_config": {
            "provider": "Groq",
            "model_name": "Llama 3.1 8B",
            "model_id": "llama3-8b-8192",
            "api_url": "https://api.groq.com/openai/v1/chat/completions",
            "temperature": 0.1,
            "max_tokens": 500
        },
        "system_prompt": "You are a helpful assistant specializing in Visage PACS support. Answer questions based on the provided context from Visage PACS documentation. Provide clear, accurate answers focused on PACS administration, configuration, troubleshooting, and usage. If the context doesn't contain enough information, say so clearly.",
        "role_based_examples": {
            "admin": [
                "What are the system requirements for Visage 7 PACS?",
                "How do I set up user permissions in Visage PACS?",
                "How do I configure PACS network settings?"
            ],
            "tech": [
                "How do I troubleshoot PACS login issues?",
                "What should I do if images are not loading?",
                "How do I configure DICOM settings?"
            ],
            "user": [
                "How do I search for studies in Visage PACS?",
                "How do I use the Visage PACS web viewer?",
                "How do I export studies from Visage PACS?"
            ]
        }
    }

# Load configuration
config = load_app_config()

# Page config
st.set_page_config(
    page_title=config["app_info"]["title"],
    page_icon=config["app_info"]["icon"],
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Cache processed knowledge base data
@st.cache_data
def load_processed_knowledge_base():
    """Load pre-processed knowledge base data instantly"""
    processed_dir = "visage_processed"
    
    if not os.path.exists(processed_dir):
        st.error(f"""
        🚨 **Processed Visage knowledge base not found!**
        
        The processed knowledge base directory '{processed_dir}' is missing.
        This should contain the pre-processed Visage PACS documentation files.
        
        Please ensure the visage_processed folder is uploaded to your deployment.
        """)
        st.stop()
    
    try:
        # Load chunks and metadata
        with open(os.path.join(processed_dir, "processed_chunks.json"), 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        # Load vectorizer
        with open(os.path.join(processed_dir, "tfidf_vectorizer.pkl"), 'rb') as f:
            vectorizer = pickle.load(f)
        
        # Load TF-IDF matrix
        matrix_data = np.load(os.path.join(processed_dir, "tfidf_matrix.npz"))
        tfidf_matrix = csr_matrix((matrix_data['data'], matrix_data['indices'], matrix_data['indptr']), 
                                 shape=matrix_data['shape'])
        
        # Load metadata
        with open(os.path.join(processed_dir, "metadata.json"), 'r') as f:
            metadata = json.load(f)
        
        return {
            'chunks': chunks_data['chunks'],
            'documents': chunks_data['documents'],
            'vectorizer': vectorizer,
            'tfidf_matrix': tfidf_matrix,
            'metadata': metadata,
            'loaded_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        st.error(f"""
        🚨 **Error loading processed Visage knowledge base:**
        
        {str(e)}
        
        Please ensure all processed files are properly uploaded to the visage_processed folder:
        - processed_chunks.json
        - tfidf_vectorizer.pkl
        - tfidf_matrix.npz
        - metadata.json
        """)
        st.stop()

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'messages': [],
        'usage_stats': {'requests_today': 0, 'total_requests': 0},
        'kb_loaded_at': None
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

def search_documents(query: str, kb_data: dict, max_results: int = 5) -> List[Dict]:
    """Search for relevant document chunks using pre-loaded TF-IDF"""
    if not kb_data['chunks'] or kb_data['vectorizer'] is None:
        return []
    
    # Transform query using pre-loaded vectorizer
    query_vector = kb_data['vectorizer'].transform([query])
    
    # Calculate similarities using pre-computed matrix
    similarities = cosine_similarity(query_vector, kb_data['tfidf_matrix']).flatten()
    
    # Get top results
    top_indices = similarities.argsort()[-max_results:][::-1]
    
    results = []
    for idx in top_indices:
        if similarities[idx] > 0.1:  # Minimum similarity threshold
            chunk = kb_data['chunks'][idx]
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
        headers["HTTP-Referer"] = "https://visage-pacs-support-hub.streamlit.app"
        headers["X-Title"] = "Visage PACS Support Hub"
    
    messages = [
        {
            "role": "system",
            "content": config["system_prompt"]
        },
        {
            "role": "user",
            "content": f"Context from Visage PACS documentation:\n\n{context}\n\nQuestion: {prompt}\n\nPlease provide a clear, helpful answer based on the context above."
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

def display_knowledge_base_info(kb_data: dict):
    """Display knowledge base loading status"""
    with st.expander("📚 Visage PACS Knowledge Base Status", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("PACS Documents Loaded", len(kb_data['documents']))
            st.metric("Text Chunks", len(kb_data['chunks']))
            
        with col2:
            st.metric("Search Index Size", f"{kb_data['tfidf_matrix'].shape[0]}×{kb_data['tfidf_matrix'].shape[1]}")
            processed_time = datetime.fromisoformat(kb_data['metadata']['processed_at'])
            st.caption(f"**Processed:** {processed_time.strftime('%Y-%m-%d %H:%M')}")
        
        # Document list
        st.markdown("**📄 Available Visage PACS Documentation:**")
        for doc in kb_data['documents']:
            st.markdown(f"• **{doc['filename']}** ({doc['chunk_count']} chunks)")

def main():
    # Initialize session state
    init_session_state()
    
    # Load knowledge base (cached) with detailed error handling
    with st.spinner("⚡ Loading Visage PACS Support Hub..."):
        try:
            kb_data = load_processed_knowledge_base()
            if kb_data is None:
                st.error("🚨 **CRITICAL ERROR:** Processed knowledge base returned None")
                st.error("This should not happen if files exist. Check app logs.")
                st.stop()
            st.session_state.kb_loaded_at = kb_data['loaded_at']
            st.success("✅ **Visage PACS knowledge base loaded instantly!** Ready for questions.")
        except Exception as e:
            st.error(f"🚨 **FATAL ERROR loading knowledge base:** {str(e)}")
            st.error("The app cannot continue without the processed knowledge base.")
            st.stop()
    
    # App header
    st.title(f"{config['app_info']['icon']} {config['app_info']['title']}")
    st.markdown(config["app_info"]["description"])
    
    # Knowledge base info
    display_knowledge_base_info(kb_data)
    
    # Role-based quick access
    with st.expander("🎯 Quick Start by Role", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**👨‍💼 PACS Administrators**")
            for question in config.get("role_based_examples", {}).get("admin", []):
                if st.button(question, key=f"admin_{question}", use_container_width=True):
                    st.session_state.quick_question = question
                    st.rerun()
        
        with col2:
            st.markdown("**🔧 Technical Support**")
            for question in config.get("role_based_examples", {}).get("tech", []):
                if st.button(question, key=f"tech_{question}", use_container_width=True):
                    st.session_state.quick_question = question
                    st.rerun()
        
        with col3:
            st.markdown("**👥 PACS Users**")
            for question in config.get("role_based_examples", {}).get("user", []):
                if st.button(question, key=f"user_{question}", use_container_width=True):
                    st.session_state.quick_question = question
                    st.rerun()
    
    # Check model availability
    if not is_model_available():
        st.error(f"⚠️ {config['model_config']['provider']} API key is not configured. Please add it in Streamlit secrets.")
        st.info(f"Add `{config['model_config']['provider'].upper()}_API_KEY` to your Streamlit secrets.")
        st.stop()
    
    # Main chat interface
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.header("💬 Visage PACS Support")
    
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
        with st.spinner("Searching Visage documentation and generating answer..."):
            # Search for relevant chunks
            search_results = search_documents(prompt, kb_data, max_results=5)
            
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
                response = "I couldn't find relevant information in the Visage PACS documentation to answer that question. Please try rephrasing your question or contact Visage technical support."
                st.session_state.messages.append({"role": "assistant", "content": response})
        
        st.rerun()
    
    # Chat input
    if prompt := st.chat_input("Ask about PACS configuration, troubleshooting, user management, or system administration..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate response
        with st.spinner("Searching Visage documentation and generating answer..."):
            # Search for relevant chunks
            search_results = search_documents(prompt, kb_data, max_results=5)
            
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
                response = "I couldn't find relevant information in the Visage PACS documentation to answer that question. Please try rephrasing your question or contact Visage technical support."
                st.session_state.messages.append({"role": "assistant", "content": response})
        
        st.rerun()

    # Footer info
    st.markdown("---")
    
    # Support information
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("**📋 Visage PACS Support Escalation:**")
        st.markdown("• **Complex PACS configuration** → Contact Visage technical support")
        st.markdown("• **DICOM integration issues** → Consult your PACS administrator")
        st.markdown("• **System downtime/critical issues** → Follow emergency procedures")
    
    with col2:
        with st.expander("📊 Performance Stats"):
            st.metric("Load Time", "⚡ Instant")
            st.metric("Questions Asked", len([m for m in st.session_state.messages if m["role"] == "user"]))
            status = "🟢 Ready" if is_model_available() else "🔴 API Issue"
            st.caption(f"**Status:** {status}")
            
            if kb_data:
                kb_time = datetime.fromisoformat(kb_data['metadata']['processed_at'])
                st.caption(f"**KB Updated:** {kb_time.strftime('%m/%d/%Y')}")

if __name__ == "__main__":
    main()