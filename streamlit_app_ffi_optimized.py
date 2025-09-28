#!/usr/bin/env python3
"""
FFI Support Hub - Optimized Version
Ultra-fast loading with pre-processed knowledge base
"""

import streamlit as st
import os
import json
import pickle
import time
import glob
import re
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import requests
import pdfplumber

# Load app configuration
@st.cache_data
def load_app_config():
    """Load app configuration from JSON file"""
    try:
        with open('app_config_ffi.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("App configuration file not found. Please ensure app_config_ffi.json exists.")
        st.stop()
    except json.JSONDecodeError:
        st.error("Invalid app configuration file. Please check app_config_ffi.json format.")
        st.stop()

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
    processed_dir = "processed_knowledge_base_ffi"
    
    if not os.path.exists(processed_dir):
        st.error(f"""
        🚨 **Processed knowledge base not found!**
        
        The processed knowledge base directory '{processed_dir}' is missing.
        Please run the preprocessing script first:
        
        ```bash
        python preprocess_knowledge_base_ffi.py
        ```
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
        🚨 **Error loading processed knowledge base:**
        
        {str(e)}
        
        Please run the preprocessing script to regenerate:
        ```bash
        python preprocess_knowledge_base_ffi.py
        ```
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

@st.cache_data
def extract_text_from_pdf_live(file_path: str) -> Optional[str]:
    """Extract text from PDF file for live processing"""
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
    except Exception as e:
        st.warning(f"Error reading {os.path.basename(file_path)}: {str(e)}")
        return None
    return text.strip()

def chunk_text_live(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks for live processing"""
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

def search_live_pdfs(query: str, knowledge_base_dir: str = "knowledge_base_ffi", max_results: int = 3) -> List[Dict]:
    """Search original PDFs for enhanced accuracy"""
    if not os.path.exists(knowledge_base_dir):
        return []
    
    pdf_files = glob.glob(os.path.join(knowledge_base_dir, "**/*.pdf"), recursive=True)
    live_results = []
    
    # Create a simple TF-IDF for live search
    all_chunks = []
    chunk_metadata = []
    
    for pdf_file in pdf_files:
        text = extract_text_from_pdf_live(pdf_file)
        if text:
            chunks = chunk_text_live(text)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    'filename': os.path.basename(pdf_file),
                    'chunk_index': i,
                    'text': chunk
                })
    
    if not all_chunks:
        return []
    
    try:
        # Quick TF-IDF search on live chunks
        vectorizer = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(all_chunks)
        query_vector = vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        
        # Get top results
        top_indices = similarities.argsort()[-max_results:][::-1]
        
        for idx in top_indices:
            if similarities[idx] > 0.1:
                chunk_info = chunk_metadata[idx]
                live_results.append({
                    'text': chunk_info['text'],
                    'source': chunk_info['filename'],
                    'similarity': float(similarities[idx]),
                    'chunk_index': chunk_info['chunk_index'],
                    'source_type': 'live_pdf'
                })
    except Exception as e:
        st.warning(f"Live PDF search error: {str(e)}")
    
    return live_results

def search_documents(query: str, kb_data: dict, max_results: int = 5, use_hybrid: bool = True, confidence_threshold: float = 0.3) -> List[Dict]:
    """Hybrid search: Fast preprocessed + Enhanced live PDF search"""
    if not kb_data['chunks'] or kb_data['vectorizer'] is None:
        return []
    
    # STEP 1: Fast preprocessed search
    query_vector = kb_data['vectorizer'].transform([query])
    similarities = cosine_similarity(query_vector, kb_data['tfidf_matrix']).flatten()
    top_indices = similarities.argsort()[-max_results:][::-1]
    
    preprocessed_results = []
    max_similarity = 0
    
    for idx in top_indices:
        if similarities[idx] > 0.1:
            chunk = kb_data['chunks'][idx]
            similarity = float(similarities[idx])
            max_similarity = max(max_similarity, similarity)
            preprocessed_results.append({
                'text': chunk['text'],
                'source': chunk['filename'],
                'similarity': similarity,
                'chunk_index': chunk['chunk_index'],
                'source_type': 'preprocessed'
            })
    
    # STEP 2: Enhanced search if confidence is low or user wants hybrid
    if use_hybrid and (max_similarity < confidence_threshold or len(preprocessed_results) < 3):
        with st.spinner("🔍 Enhancing search with live PDF analysis..."):
            live_results = search_live_pdfs(query, max_results=3)
            
            # Combine and deduplicate results
            all_results = preprocessed_results + live_results
            
            # Remove duplicates based on text similarity
            unique_results = []
            for result in all_results:
                is_duplicate = False
                for existing in unique_results:
                    if (result['source'] == existing['source'] and 
                        abs(result['chunk_index'] - existing['chunk_index']) <= 1):
                        # Keep the higher similarity result
                        if result['similarity'] > existing['similarity']:
                            unique_results.remove(existing)
                        else:
                            is_duplicate = True
                        break
                if not is_duplicate:
                    unique_results.append(result)
            
            # Sort by similarity and return top results
            unique_results.sort(key=lambda x: x['similarity'], reverse=True)
            return unique_results[:max_results]
    
    return preprocessed_results

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
        headers["HTTP-Referer"] = "https://inteleorchestrator-support-hub.streamlit.app"
        headers["X-Title"] = "InteleOrchestrator Support Hub"
    
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

def display_knowledge_base_info(kb_data: dict):
    """Display knowledge base loading status with hybrid search info"""
    with st.expander("📚 Hybrid Knowledge Base Status", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📋 Preprocessed Docs", len(kb_data['documents']))
            st.metric("⚡ Fast Chunks", len(kb_data['chunks']))
            
        with col2:
            # Check live PDF availability
            live_pdf_dir = "knowledge_base_ffi"
            live_pdfs = []
            if os.path.exists(live_pdf_dir):
                live_pdfs = glob.glob(os.path.join(live_pdf_dir, "**/*.pdf"), recursive=True)
            st.metric("📄 Live PDFs", len(live_pdfs))
            st.metric("🔍 Search Index", f"{kb_data['tfidf_matrix'].shape[0]}×{kb_data['tfidf_matrix'].shape[1]}")
            
        with col3:
            processed_time = datetime.fromisoformat(kb_data['metadata']['processed_at'])
            st.caption(f"**Preprocessed:** {processed_time.strftime('%Y-%m-%d %H:%M')}")
            
            # Search mode indicator
            if len(live_pdfs) > 0:
                st.success("🔄 **Hybrid Mode Active**")
                st.caption("Fast + Enhanced search")
            else:
                st.info("⚡ **Fast Mode Only**")
                st.caption("Preprocessed search")
        
        # Document comparison
        st.markdown("**📄 Document Coverage:**")
        preprocessed_docs = {doc['filename'] for doc in kb_data['documents']}
        live_docs = {os.path.basename(pdf) for pdf in live_pdfs}
        
        all_docs = preprocessed_docs.union(live_docs)
        for doc in sorted(all_docs):
            icons = []
            if doc in preprocessed_docs:
                icons.append("⚡")
            if doc in live_docs:
                icons.append("📄")
            st.markdown(f"• {''.join(icons)} **{doc}**")
        
        if len(live_pdfs) > 0:
            st.info("💡 **Hybrid Search**: Starts with fast preprocessed search, enhances with live PDF analysis when needed for better accuracy.")

def main():
    # Initialize session state
    init_session_state()
    
    # Load knowledge base (cached) with detailed error handling
    with st.spinner("⚡ Loading InteleOrchestrator Support Hub..."):
        try:
            kb_data = load_processed_knowledge_base()
            if kb_data is None:
                st.error("🚨 **CRITICAL ERROR:** Processed knowledge base returned None")
                st.error("This should not happen if files exist. Check app logs.")
                st.stop()
            st.session_state.kb_loaded_at = kb_data['loaded_at']
            st.success("✅ **Knowledge base loaded instantly!** Ready for questions.")
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
    
    # Check model availability
    if not is_model_available():
        st.error(f"⚠️ {config['model_config']['provider']} API key is not configured. Please add it in Streamlit secrets.")
        st.info(f"Add `{config['model_config']['provider'].upper()}_API_KEY` to your Streamlit secrets.")
        st.stop()
    
    # Search mode controls
    with st.expander("🔧 Search Settings", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            use_hybrid = st.checkbox("🔄 Enhanced Search", value=True, 
                                   help="Uses live PDF analysis for better accuracy when confidence is low")
        with col2:
            confidence_threshold = st.slider("Confidence Threshold", 0.1, 0.5, 0.3, 0.05,
                                           help="Lower values trigger enhanced search more often")
        with col3:
            force_full_search = st.checkbox("🚀 Force Full Search", value=False,
                                           help="Skip fast search entirely and use intensive live PDF analysis for all queries")
        
        # Override settings when force full search is enabled
        if force_full_search:
            use_hybrid = True  # Force hybrid mode
            confidence_threshold = 1.0  # Set threshold to max to always trigger enhanced search
            st.info("🚀 **Full Search Mode Active** - Using intensive live PDF analysis for maximum accuracy")
    
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
                
                # Show sources with search type indicators
                if "sources" in message and message["sources"]:
                    with st.expander("📖 Sources", expanded=False):
                        for i, source in enumerate(message["sources"][:3]):
                            # Source type indicator
                            source_icon = "⚡" if source.get('source_type') == 'preprocessed' else "📄"
                            source_label = "Fast" if source.get('source_type') == 'preprocessed' else "Enhanced"
                            
                            st.markdown(f"{source_icon} **{source['source']}** (Relevance: {source['similarity']:.1%}) - *{source_label}*")
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
        if force_full_search:
            search_spinner_text = "🚀 Full search mode - analyzing all PDFs intensively..."
        elif use_hybrid:
            search_spinner_text = "🔍 Smart searching (fast + enhanced)..."
        else:
            search_spinner_text = "🔍 Searching documentation..."
        
        with st.spinner(search_spinner_text):
            # Search for relevant chunks
            search_results = search_documents(prompt, kb_data, max_results=5, use_hybrid=use_hybrid, confidence_threshold=confidence_threshold)
            
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
        if force_full_search:
            search_spinner_text = "🚀 Full search mode - analyzing all PDFs intensively..."
        elif use_hybrid:
            search_spinner_text = "🔍 Smart searching (fast + enhanced)..."
        else:
            search_spinner_text = "🔍 Searching documentation..."
        
        with st.spinner(search_spinner_text):
            # Search for relevant chunks
            search_results = search_documents(prompt, kb_data, max_results=5, use_hybrid=use_hybrid, confidence_threshold=confidence_threshold)
            
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
        with st.expander("📊 Performance Stats"):
            st.metric("Load Time", "⚡ Instant")
            st.metric("Questions Asked", len([m for m in st.session_state.messages if m["role"] == "user"]))
            
            # Search mode status
            live_pdf_dir = "knowledge_base_ffi"
            has_live_pdfs = os.path.exists(live_pdf_dir) and len(glob.glob(os.path.join(live_pdf_dir, "**/*.pdf"), recursive=True)) > 0
            
            if force_full_search:
                search_mode = "🚀 Full Search (Forced)"
            elif use_hybrid and has_live_pdfs:
                search_mode = "🔄 Hybrid"
            else:
                search_mode = "⚡ Fast Only"
            st.caption(f"**Search Mode:** {search_mode}")
            
            status = "🟢 Ready" if is_model_available() else "🔴 API Issue"
            st.caption(f"**API Status:** {status}")
            
            if kb_data:
                kb_time = datetime.fromisoformat(kb_data['metadata']['processed_at'])
                st.caption(f"**KB Updated:** {kb_time.strftime('%m/%d/%Y')}")
                
            # Enhanced search indicator
            if force_full_search:
                st.caption("**Mode:** Full intensive PDF analysis")
            elif use_hybrid and has_live_pdfs:
                st.caption(f"**Enhanced Trigger:** {confidence_threshold:.1%} confidence")

if __name__ == "__main__":
    main()