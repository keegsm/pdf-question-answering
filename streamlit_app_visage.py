#!/usr/bin/env python3
"""
Visage PACS Support Hub - GitHub Version
Ultra-fast loading with pre-processed knowledge base for Visage PACS documentation
Optimized for GitHub repository structure
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
    try:
        with open('visage_config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Return default Visage configuration if config file doesn't exist
        return {
            "app_info": {
                "title": "Visage PACS Support Hub",
                "icon": "📡",
                "version": "1.0.0",
                "description": "AI-powered support for Visage PACS documentation and troubleshooting"
            },
            "branding": {
                "primary_color": "#1f4e79",
                "secondary_color": "#2e6da4",
                "accent_color": "#337ab7",
                "company_name": "Visage Imaging",
                "product_suite": "Visage PACS"
            },
            "knowledge_base": {
                "processed_folder": "visage_processed"
            },
            "llm_settings": {
                "default_model": "groq",
                "temperature": 0.1,
                "max_tokens": 500,
                "context_chunks": 3
            }
        }
    except json.JSONDecodeError:
        st.error("Invalid app configuration file. Please check visage_config.json format.")
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
    processed_dir = config["knowledge_base"]["processed_folder"]
    
    if not os.path.exists(processed_dir):
        st.error(f"""
        🚨 **Processed knowledge base not found!**
        
        The processed knowledge base directory '{processed_dir}' is missing.
        Please ensure the visage_processed folder with processed files is uploaded to GitHub.
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
        
        Please ensure all processed files are properly uploaded to the visage_processed folder.
        """)
        st.stop()

# Search function
def search_knowledge_base(query: str, kb_data: Dict, top_k: int = 5) -> List[Dict]:
    """Search the knowledge base using TF-IDF similarity"""
    if not query.strip():
        return []
    
    try:
        # Transform query using the fitted vectorizer
        query_vector = kb_data['vectorizer'].transform([query.lower()])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, kb_data['tfidf_matrix']).flatten()
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.01:  # Minimum similarity threshold
                chunk = kb_data['chunks'][idx]
                results.append({
                    'text': chunk['text'],
                    'filename': chunk['filename'],
                    'chunk_index': chunk['chunk_index'],
                    'similarity': float(similarities[idx])
                })
        
        return results
        
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []

# LLM API calls
def call_llm_api(prompt: str, context: str) -> str:
    """Call LLM API with fallback options"""
    
    # Try Groq first (free tier)
    if os.getenv("GROQ_API_KEY"):
        try:
            headers = {
                "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "llama3-8b-8192",
                "messages": [
                    {
                        "role": "system",
                        "content": f"You are a helpful assistant specializing in {config['branding']['product_suite']} support. Answer questions based on the provided context from Visage documentation. If the context doesn't contain enough information, say so clearly."
                    },
                    {
                        "role": "user", 
                        "content": f"Context from Visage documentation:\n{context}\n\nQuestion: {prompt}\n\nPlease provide a clear, accurate answer based on the Visage documentation provided."
                    }
                ],
                "temperature": config["llm_settings"]["temperature"],
                "max_tokens": config["llm_settings"]["max_tokens"]
            }
            
            response = requests.post("https://api.groq.com/openai/v1/chat/completions", 
                                   headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
                
        except Exception as e:
            st.warning(f"Groq API error: {e}")
    
    # Try OpenAI as fallback
    if os.getenv("OPENAI_API_KEY"):
        try:
            headers = {
                "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {
                        "role": "system",
                        "content": f"You are a helpful assistant specializing in {config['branding']['product_suite']} support. Answer questions based on the provided context from Visage documentation."
                    },
                    {
                        "role": "user",
                        "content": f"Context: {context}\n\nQuestion: {prompt}"
                    }
                ],
                "temperature": config["llm_settings"]["temperature"],
                "max_tokens": config["llm_settings"]["max_tokens"]
            }
            
            response = requests.post("https://api.openai.com/v1/chat/completions",
                                   headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
                
        except Exception as e:
            st.warning(f"OpenAI API error: {e}")
    
    # Fallback response
    return f"""I found relevant information in the {config['branding']['product_suite']} documentation, but I need an API key to provide a detailed answer. 

Based on the search results, here are the relevant sections:

{context[:500]}...

Please configure your GROQ_API_KEY or OPENAI_API_KEY environment variable to get detailed AI-powered answers."""

# Load knowledge base
@st.cache_data
def get_knowledge_base():
    return load_processed_knowledge_base()

# Load demo questions
@st.cache_data
def load_demo_questions():
    """Load demo questions from JSON file"""
    try:
        with open('visage_demo_questions.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "demo_questions": [
                "What are the system requirements for Visage 7 PACS?",
                "How do I set up user permissions in Visage PACS?",
                "How do I configure DICOM settings in Visage PACS?",
                "How do I search for studies in Visage PACS?",
                "What should I do if images are not loading in PACS?",
                "How do I troubleshoot PACS login issues?"
            ]
        }

def main():
    # Custom CSS with Visage branding
    st.markdown(f"""
    <style>
    .main-header {{
        background: linear-gradient(90deg, {config['branding']['primary_color']}, {config['branding']['secondary_color']});
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }}
    .main-header h1 {{
        margin: 0;
        color: white !important;
    }}
    .main-header p {{
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }}
    .stTextInput > div > div > input {{
        font-size: 16px;
    }}
    .search-result {{
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid {config['branding']['accent_color']};
    }}
    .source-info {{
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.5rem;
    }}
    .stats-container {{
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }}
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown(f"""
    <div class="main-header">
        <h1>{config['app_info']['icon']} {config['app_info']['title']}</h1>
        <p>{config['app_info']['description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load knowledge base
    with st.spinner("🚀 Loading Visage PACS knowledge base..."):
        kb_data = get_knowledge_base()
    
    # Knowledge base stats
    st.markdown(f"""
    <div class="stats-container">
        <strong>📊 Knowledge Base Status:</strong><br>
        📁 Documents: {len(kb_data['documents'])} Visage PACS manuals and guides<br>
        📝 Text chunks: {len(kb_data['chunks'])} searchable sections<br>
        🔍 Search index: {kb_data['tfidf_matrix'].shape[1]} features<br>
        ⏰ Last updated: {kb_data['metadata'].get('processed_at', 'Unknown')}
    </div>
    """, unsafe_allow_html=True)
    
    # Document list
    with st.expander("📋 Available Visage PACS Documentation", expanded=False):
        for doc in kb_data['documents']:
            st.write(f"**{doc['filename']}** - {doc['chunk_count']} sections ({doc['text_length']:,} characters)")
    
    # Main search interface
    st.markdown("### 🔍 Ask about Visage PACS")
    
    # Load and display demo questions
    demo_data = load_demo_questions()
    example_questions = demo_data.get("demo_questions", [])
    
    selected_example = st.selectbox("💡 Try an example question:", [""] + example_questions)
    
    # Search input
    query = st.text_input("Or type your own question:", value=selected_example, 
                         placeholder="e.g., How do I configure DICOM settings in Visage PACS?")
    
    if query:
        with st.spinner("🔎 Searching Visage PACS documentation..."):
            # Search knowledge base
            search_results = search_knowledge_base(query, kb_data, top_k=config["llm_settings"]["context_chunks"])
            
            if search_results:
                # Create context from top results
                context = "\n\n".join([result['text'] for result in search_results[:config["llm_settings"]["context_chunks"]]])
                
                # Get AI answer
                with st.spinner("🤖 Generating answer..."):
                    ai_answer = call_llm_api(query, context)
                
                # Display AI answer
                st.markdown("### 🤖 AI Answer")
                st.markdown(f"""
                <div style="background: #e8f5e8; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #28a745;">
                {ai_answer}
                </div>
                """, unsafe_allow_html=True)
                
                # Display sources
                st.markdown("### 📖 Sources from Visage PACS Documentation")
                for i, result in enumerate(search_results, 1):
                    similarity_percent = int(result['similarity'] * 100)
                    st.markdown(f"""
                    <div class="search-result">
                        <strong>📄 Source {i} - {result['filename']}</strong> 
                        <span style="color: #28a745;">({similarity_percent}% match)</span>
                        <p>{result['text'][:400]}{"..." if len(result['text']) > 400 else ""}</p>
                        <div class="source-info">
                            Section {result['chunk_index'] + 1} | Relevance: {similarity_percent}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("🤔 No relevant information found in the Visage PACS documentation for your query. Try rephrasing your question or using different keywords.")
    
    # Sidebar with API configuration
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # API status
        groq_configured = bool(os.getenv("GROQ_API_KEY"))
        openai_configured = bool(os.getenv("OPENAI_API_KEY"))
        
        st.markdown("**🔑 API Status:**")
        st.markdown(f"{'✅' if groq_configured else '❌'} Groq API (Free)")
        st.markdown(f"{'✅' if openai_configured else '❌'} OpenAI API (Paid)")
        
        if not groq_configured and not openai_configured:
            st.warning("⚠️ No API keys configured. Answers will be limited.")
            st.markdown("""
            Add to Streamlit secrets:
            ```
            GROQ_API_KEY = "your_key_here"
            ```
            """)
        
        st.markdown("---")
        st.markdown(f"**📱 App Version:** {config['app_info']['version']}")
        st.markdown(f"**🏢 Product:** {config['branding']['product_suite']}")
        st.markdown("**📊 Knowledge Base:** PACS-focused (Lite)")

if __name__ == "__main__":
    main()