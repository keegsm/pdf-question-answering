#!/usr/bin/env python3
"""
Simplified Streamlit PDF Question-Answering App
Works perfectly on Streamlit Cloud without ChromaDB
"""

import streamlit as st
import os
import uuid
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
import requests
import json
import re

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
if 'document_chunks' not in st.session_state:
    st.session_state.document_chunks = []
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'tfidf_matrix' not in st.session_state:
    st.session_state.tfidf_matrix = None

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    """Split text into overlapping chunks"""
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

def update_search_index():
    """Update the TF-IDF search index"""
    if not st.session_state.document_chunks:
        return
    
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

def search_documents(query, max_results=5):
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
            "content": "You are a helpful assistant that answers questions based on provided context. Be concise and accurate. If the context doesn't contain enough information, say so clearly."
        },
        {
            "role": "user",
            "content": f"Context from documents:\n\n{context}\n\nQuestion: {prompt}\n\nPlease provide a clear, helpful answer based on the context above."
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
        st.error(f"Groq API error: {str(e)}")
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
            "content": "You are a helpful assistant that answers questions based on provided context. Be concise and accurate. If the context doesn't contain enough information, say so clearly."
        },
        {
            "role": "user",
            "content": f"Context from documents:\n\n{context}\n\nQuestion: {prompt}\n\nPlease provide a clear, helpful answer based on the context above."
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
        st.error(f"OpenAI API error: {str(e)}")
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
    return "I apologize, but I couldn't access the AI service to answer your question. Please check your API key configuration."

def main():
    st.title("📄 PDF Question Answering System")
    st.markdown("Upload PDF documents and ask questions about their content using AI")
    
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
                # Check if file already processed
                if not any(doc['filename'] == uploaded_file.name for doc in st.session_state.documents):
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        text = extract_text_from_pdf(uploaded_file)
                        if text and len(text.strip()) > 100:
                            # Chunk the text
                            chunks = chunk_text(text)
                            
                            if chunks:
                                # Add document info
                                doc_info = {
                                    'id': str(uuid.uuid4()),
                                    'filename': uploaded_file.name,
                                    'chunks': len(chunks),
                                    'text_length': len(text)
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
                                
                                st.success(f"✅ {uploaded_file.name} processed! ({len(chunks)} chunks)")
                            else:
                                st.error(f"Could not extract meaningful content from {uploaded_file.name}")
                        else:
                            st.error(f"No text found in {uploaded_file.name}")
        
        # Show uploaded documents
        st.header("📚 Documents")
        if st.session_state.documents:
            for doc in st.session_state.documents:
                st.text(f"📄 {doc['filename']}")
                st.caption(f"   {doc['chunks']} chunks • {doc['text_length']:,} chars")
                
                # Delete button
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
                # Add to chat input
                st.session_state.demo_question = question
                st.rerun()
    
    # Main chat area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("💬 Ask Questions")
    
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
                if "sources" in message and message["sources"]:
                    with st.expander("📖 Sources", expanded=False):
                        for i, source in enumerate(message["sources"][:3]):
                            st.markdown(f"**📄 {source['source']}** (Relevance: {source['similarity']:.1%})")
                            st.caption(source['text'][:300] + ("..." if len(source['text']) > 300 else ""))
                            if i < len(message["sources"]) - 1:
                                st.divider()
    
    # Handle demo question
    demo_input = ""
    if hasattr(st.session_state, 'demo_question'):
        demo_input = st.session_state.demo_question
        delattr(st.session_state, 'demo_question')
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents...", value=demo_input):
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
                    response = get_llm_response(prompt, context)
                    
                    # Add to message history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "sources": search_results
                    })
                else:
                    response = "I couldn't find relevant information in your uploaded documents to answer that question."
                    st.session_state.messages.append({"role": "assistant", "content": response})
        
        st.rerun()

    # Footer info
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Documents", len(st.session_state.documents))
    with col2:
        st.metric("Text Chunks", len(st.session_state.document_chunks))
    with col3:
        st.metric("Chat Messages", len(st.session_state.messages))

if __name__ == "__main__":
    main()