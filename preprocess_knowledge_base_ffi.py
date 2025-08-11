#!/usr/bin/env python3
"""
Knowledge Base Preprocessing Script
Processes PDFs offline to eliminate runtime delays in Streamlit app
"""

import os
import json
import pickle
import time
import glob
import re
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pdfplumber

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
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

def extract_text_from_pdf(file_path: str) -> Optional[str]:
    """Extract text from PDF file"""
    text = ""
    try:
        print(f"  Processing {os.path.basename(file_path)}...")
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"    {total_pages} pages")
            
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
                
                # Progress indicator
                if (page_num + 1) % 10 == 0 or page_num + 1 == total_pages:
                    print(f"    Processed {page_num + 1}/{total_pages} pages")
                
    except Exception as e:
        print(f"    ERROR: {str(e)}")
        return None
    
    return text.strip()

def preprocess_knowledge_base(knowledge_base_dir: str = "knowledge_base_ffi", 
                            output_dir: str = "processed_knowledge_base_ffi"):
    """Preprocess all documents in knowledge base directory"""
    
    print("🔄 InteleOrchestrator Knowledge Base Preprocessing")
    print("=" * 55)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all PDF files
    pdf_files = glob.glob(os.path.join(knowledge_base_dir, "*.pdf"))
    
    if not pdf_files:
        print(f"❌ No PDF files found in {knowledge_base_dir}")
        return False
    
    print(f"📁 Found {len(pdf_files)} PDF files")
    print()
    
    # Process each document
    all_chunks = []
    document_info = []
    
    start_time = time.time()
    
    for doc_index, file_path in enumerate(pdf_files, 1):
        filename = os.path.basename(file_path)
        print(f"📄 Document {doc_index}/{len(pdf_files)}: {filename}")
        
        # Extract text
        text = extract_text_from_pdf(file_path)
        if not text or len(text.strip()) < 100:
            print(f"    ⚠️  No meaningful text found")
            continue
        
        # Chunk the text
        print(f"    Chunking text ({len(text):,} characters)...")
        chunks = chunk_text(text)
        
        if not chunks:
            print(f"    ⚠️  No chunks created")
            continue
        
        print(f"    ✅ Created {len(chunks)} chunks")
        
        # Store document info
        doc_info = {
            'filename': filename,
            'text_length': len(text),
            'chunk_count': len(chunks),
            'processed_at': datetime.now().isoformat()
        }
        document_info.append(doc_info)
        
        # Add chunks to collection
        for i, chunk in enumerate(chunks):
            chunk_info = {
                'id': f"{filename}_chunk_{i}",
                'filename': filename,
                'chunk_index': i,
                'text': chunk
            }
            all_chunks.append(chunk_info)
        
        print()
    
    if not all_chunks:
        print("❌ No chunks created from any documents")
        return False
    
    print(f"📊 Total chunks created: {len(all_chunks)}")
    print()
    
    # Build TF-IDF vectorizer and matrix
    print("🔍 Building search index...")
    chunk_texts = [chunk['text'] for chunk in all_chunks]
    
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2),
        lowercase=True
    )
    
    tfidf_matrix = vectorizer.fit_transform(chunk_texts)
    print(f"    ✅ TF-IDF matrix: {tfidf_matrix.shape}")
    
    # Save processed data
    print()
    print("💾 Saving processed data...")
    
    # Save chunks as JSON
    chunks_file = os.path.join(output_dir, "processed_chunks.json")
    with open(chunks_file, 'w', encoding='utf-8') as f:
        json.dump({
            'chunks': all_chunks,
            'documents': document_info,
            'processed_at': datetime.now().isoformat(),
            'total_chunks': len(all_chunks),
            'total_documents': len(pdf_files)
        }, f, indent=2, ensure_ascii=False)
    print(f"    ✅ Chunks saved: {chunks_file}")
    
    # Save vectorizer
    vectorizer_file = os.path.join(output_dir, "tfidf_vectorizer.pkl")
    with open(vectorizer_file, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"    ✅ Vectorizer saved: {vectorizer_file}")
    
    # Save TF-IDF matrix (sparse format)
    matrix_file = os.path.join(output_dir, "tfidf_matrix.npz")
    np.savez_compressed(matrix_file, 
                       data=tfidf_matrix.data,
                       indices=tfidf_matrix.indices,
                       indptr=tfidf_matrix.indptr,
                       shape=tfidf_matrix.shape)
    print(f"    ✅ TF-IDF matrix saved: {matrix_file}")
    
    # Create metadata file
    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump({
            'source_directory': knowledge_base_dir,
            'processed_at': datetime.now().isoformat(),
            'processing_time_seconds': round(time.time() - start_time, 2),
            'total_documents': len(pdf_files),
            'total_chunks': len(all_chunks),
            'vectorizer_features': vectorizer.get_feature_names_out().shape[0] if hasattr(vectorizer, 'get_feature_names_out') else 'unknown',
            'tfidf_matrix_shape': tfidf_matrix.shape,
            'files_created': [
                'processed_chunks.json',
                'tfidf_vectorizer.pkl', 
                'tfidf_matrix.npz',
                'metadata.json'
            ]
        }, f, indent=2)
    print(f"    ✅ Metadata saved: {metadata_file}")
    
    processing_time = time.time() - start_time
    print()
    print("🎉 Preprocessing Complete!")
    print(f"    ⏱️  Processing time: {processing_time:.1f} seconds")
    print(f"    📄 Documents processed: {len(pdf_files)}")
    print(f"    📝 Text chunks created: {len(all_chunks)}")
    print(f"    📁 Output directory: {output_dir}")
    print()
    print("💡 The Streamlit app will now load instantly using this processed data!")
    
    return True

if __name__ == "__main__":
    success = preprocess_knowledge_base()
    if not success:
        exit(1)