#!/usr/bin/env python3
"""
InteleOrchestrator Support Hub - Diagnostic Version
Shows detailed logging of what the app is doing during startup
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

# Diagnostic logging function
def log_diagnostic(message, level="INFO"):
    """Log diagnostic information to the app"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    if level == "ERROR":
        st.error(f"🔍 [{timestamp}] {message}")
    elif level == "WARNING":
        st.warning(f"🔍 [{timestamp}] {message}")
    elif level == "SUCCESS":
        st.success(f"🔍 [{timestamp}] {message}")
    else:
        st.info(f"🔍 [{timestamp}] {message}")

# Load app configuration
@st.cache_data
def load_app_config():
    """Load app configuration from JSON file"""
    try:
        log_diagnostic("Loading app configuration...")
        with open('app_config.json', 'r') as f:
            config = json.load(f)
        log_diagnostic("✅ App configuration loaded successfully")
        return config
    except FileNotFoundError:
        log_diagnostic("❌ app_config.json not found", "ERROR")
        st.stop()
    except json.JSONDecodeError:
        log_diagnostic("❌ Invalid JSON in app_config.json", "ERROR")
        st.stop()

# Load configuration
config = load_app_config()

# Page config
st.set_page_config(
    page_title=f"🔍 DIAGNOSTIC - {config['app_info']['title']}",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Detailed file system check
def check_file_system():
    """Check what files are actually available"""
    log_diagnostic("🔍 DIAGNOSTIC MODE - Checking file system...")
    
    # Check current working directory
    cwd = os.getcwd()
    log_diagnostic(f"Current working directory: {cwd}")
    
    # List all files in root
    try:
        root_files = os.listdir('.')
        log_diagnostic(f"Files in root directory: {sorted(root_files)}")
    except Exception as e:
        log_diagnostic(f"Error listing root directory: {e}", "ERROR")
    
    # Check for processed_knowledge_base directory
    processed_dir = "processed_knowledge_base"
    if os.path.exists(processed_dir):
        log_diagnostic(f"✅ Found {processed_dir} directory", "SUCCESS")
        try:
            processed_files = os.listdir(processed_dir)
            log_diagnostic(f"Files in {processed_dir}: {sorted(processed_files)}")
            
            # Check file sizes
            for file in processed_files:
                file_path = os.path.join(processed_dir, file)
                size = os.path.getsize(file_path)
                log_diagnostic(f"  {file}: {size:,} bytes")
                
        except Exception as e:
            log_diagnostic(f"Error listing {processed_dir}: {e}", "ERROR")
    else:
        log_diagnostic(f"❌ {processed_dir} directory NOT FOUND", "ERROR")
    
    # Check for knowledge_base directory (fallback)
    kb_dir = "knowledge_base"
    if os.path.exists(kb_dir):
        log_diagnostic(f"📁 Found {kb_dir} directory (fallback)")
        try:
            kb_files = os.listdir(kb_dir)
            log_diagnostic(f"Files in {kb_dir}: {sorted(kb_files)}")
        except Exception as e:
            log_diagnostic(f"Error listing {kb_dir}: {e}", "ERROR")
    else:
        log_diagnostic(f"❌ {kb_dir} directory also NOT FOUND", "ERROR")

# Cache processed knowledge base data with diagnostics
@st.cache_data
def load_processed_knowledge_base():
    """Load pre-processed knowledge base data with detailed logging"""
    log_diagnostic("🚀 Starting knowledge base loading...")
    
    processed_dir = "processed_knowledge_base"
    
    if not os.path.exists(processed_dir):
        log_diagnostic(f"❌ {processed_dir} directory not found - will fall back to runtime processing", "WARNING")
        return None
    
    log_diagnostic(f"✅ {processed_dir} directory found", "SUCCESS")
    
    try:
        # Check each required file
        required_files = [
            "processed_chunks.json",
            "tfidf_vectorizer.pkl", 
            "tfidf_matrix.npz",
            "metadata.json"
        ]
        
        for file in required_files:
            file_path = os.path.join(processed_dir, file)
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                log_diagnostic(f"✅ Found {file} ({size:,} bytes)")
            else:
                log_diagnostic(f"❌ Missing {file}", "ERROR")
                return None
        
        # Load chunks and metadata
        log_diagnostic("Loading processed chunks...")
        with open(os.path.join(processed_dir, "processed_chunks.json"), 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        log_diagnostic(f"✅ Loaded {len(chunks_data['chunks'])} chunks")
        
        # Load vectorizer
        log_diagnostic("Loading TF-IDF vectorizer...")
        with open(os.path.join(processed_dir, "tfidf_vectorizer.pkl"), 'rb') as f:
            vectorizer = pickle.load(f)
        log_diagnostic(f"✅ Loaded vectorizer with {len(vectorizer.vocabulary_)} features")
        
        # Load TF-IDF matrix
        log_diagnostic("Loading TF-IDF matrix...")
        matrix_data = np.load(os.path.join(processed_dir, "tfidf_matrix.npz"))
        tfidf_matrix = csr_matrix((matrix_data['data'], matrix_data['indices'], matrix_data['indptr']), 
                                 shape=matrix_data['shape'])
        log_diagnostic(f"✅ Loaded TF-IDF matrix: {tfidf_matrix.shape}")
        
        # Load metadata
        log_diagnostic("Loading metadata...")
        with open(os.path.join(processed_dir, "metadata.json"), 'r') as f:
            metadata = json.load(f)
        log_diagnostic(f"✅ Loaded metadata (processed at {metadata['processed_at']})")
        
        log_diagnostic("🎉 ALL PROCESSED DATA LOADED SUCCESSFULLY!", "SUCCESS")
        
        return {
            'chunks': chunks_data['chunks'],
            'documents': chunks_data['documents'],
            'vectorizer': vectorizer,
            'tfidf_matrix': tfidf_matrix,
            'metadata': metadata,
            'loaded_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        log_diagnostic(f"❌ Error loading processed data: {str(e)}", "ERROR")
        log_diagnostic("Will fall back to runtime processing", "WARNING")
        return None

def main():
    st.title("🔍 InteleOrchestrator Support Hub - DIAGNOSTIC MODE")
    st.markdown("**This version shows detailed logging to diagnose loading issues**")
    
    # File system diagnostics
    with st.expander("🗂️ File System Diagnostics", expanded=True):
        check_file_system()
    
    st.markdown("---")
    
    # Try to load processed knowledge base
    st.header("📚 Knowledge Base Loading Test")
    
    with st.spinner("Testing knowledge base loading..."):
        kb_data = load_processed_knowledge_base()
    
    if kb_data:
        st.success("🎉 OPTIMIZATION WORKING! Pre-processed data loaded successfully.")
        
        # Show what was loaded
        with st.expander("📊 Loaded Data Summary", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", len(kb_data['documents']))
                st.metric("Text Chunks", len(kb_data['chunks']))
            with col2:
                st.metric("Search Features", kb_data['tfidf_matrix'].shape[1])
                st.metric("Matrix Size", f"{kb_data['tfidf_matrix'].shape[0]}×{kb_data['tfidf_matrix'].shape[1]}")
        
        st.success("✅ **The optimized app should load instantly with this data!**")
        
    else:
        st.error("❌ OPTIMIZATION NOT WORKING - Missing processed data")
        st.warning("The app will fall back to runtime processing (30-60 seconds)")
        
        st.markdown("""
        ### 🔧 To Fix This:
        
        1. **Check if processed files are in your repository:**
           ```bash
           ls processed_knowledge_base/
           ```
        
        2. **If missing, run preprocessing locally:**
           ```bash
           python preprocess_knowledge_base.py
           git add processed_knowledge_base/
           git commit -m "Add processed knowledge base"
           git push
           ```
        
        3. **Verify Streamlit Cloud deployment includes the files**
        """)
    
    st.markdown("---")
    st.info("💡 **Once diagnostics show success, switch back to the regular optimized app for production use.**")

if __name__ == "__main__":
    main()