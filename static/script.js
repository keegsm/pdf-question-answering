/**
 * PDF Question Answering App - Frontend JavaScript
 * Handles file uploads, chat interface, and all user interactions
 */

class PDFQuestionApp {
    constructor() {
        this.init();
        this.setupEventListeners();
        this.checkSystemHealth();
        this.loadDemoQuestions();
        this.loadDocuments();
    }

    init() {
        // DOM elements
        this.elements = {
            uploadArea: document.getElementById('uploadArea'),
            fileInput: document.getElementById('fileInput'),
            uploadProgress: document.getElementById('uploadProgress'),
            progressFill: document.getElementById('progressFill'),
            progressText: document.getElementById('progressText'),
            documentsList: document.getElementById('documentsList'),
            chatMessages: document.getElementById('chatMessages'),
            chatInput: document.getElementById('chatInput'),
            sendButton: document.getElementById('sendButton'),
            charCount: document.getElementById('charCount'),
            statusDot: document.getElementById('status-dot'),
            statusText: document.getElementById('status-text'),
            demoQuestions: document.getElementById('demoQuestions'),
            toastContainer: document.getElementById('toastContainer'),
            loadingOverlay: document.getElementById('loadingOverlay'),
            confirmModal: document.getElementById('confirmModal'),
            confirmMessage: document.getElementById('confirmMessage'),
            confirmOk: document.getElementById('confirmOk'),
            confirmCancel: document.getElementById('confirmCancel'),
            modalClose: document.getElementById('modalClose')
        };

        // App state
        this.state = {
            documents: [],
            chatHistory: [],
            isUploading: false,
            isProcessing: false,
            systemHealth: null,
            demoQuestions: []
        };
    }

    setupEventListeners() {
        // File upload events
        this.elements.uploadArea.addEventListener('click', () => {
            this.elements.fileInput.click();
        });

        this.elements.fileInput.addEventListener('change', (e) => {
            this.handleFileUpload(e.target.files);
        });

        // Drag and drop events
        this.elements.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.elements.uploadArea.classList.add('dragover');
        });

        this.elements.uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            this.elements.uploadArea.classList.remove('dragover');
        });

        this.elements.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.elements.uploadArea.classList.remove('dragover');
            this.handleFileUpload(e.dataTransfer.files);
        });

        // Chat events
        this.elements.chatInput.addEventListener('input', (e) => {
            this.updateCharCount();
            this.updateSendButton();
        });

        this.elements.chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        this.elements.sendButton.addEventListener('click', () => {
            this.sendMessage();
        });

        // Modal events
        this.elements.modalClose.addEventListener('click', () => {
            this.hideModal();
        });

        this.elements.confirmCancel.addEventListener('click', () => {
            this.hideModal();
        });

        // Close modal on outside click
        this.elements.confirmModal.addEventListener('click', (e) => {
            if (e.target === this.elements.confirmModal) {
                this.hideModal();
            }
        });
    }

    // System Health Check
    async checkSystemHealth() {
        try {
            const response = await fetch('/health');
            const health = await response.json();
            this.state.systemHealth = health;
            this.updateSystemStatus(health);
        } catch (error) {
            console.error('Health check failed:', error);
            this.updateSystemStatus(null);
        }
    }

    updateSystemStatus(health) {
        if (!health) {
            this.elements.statusDot.className = 'status-dot offline';
            this.elements.statusText.textContent = 'System offline';
            return;
        }

        const hasLLM = Object.values(health.llm_backends).some(backend => backend.available);
        
        if (hasLLM) {
            this.elements.statusDot.className = 'status-dot online';
            this.elements.statusText.textContent = 'System ready';
        } else {
            this.elements.statusDot.className = 'status-dot';
            this.elements.statusText.textContent = 'No LLM available';
        }
    }

    // Demo Questions
    async loadDemoQuestions() {
        try {
            const response = await fetch('/demo');
            const data = await response.json();
            this.state.demoQuestions = data.demo_questions || [];
            this.renderDemoQuestions();
        } catch (error) {
            console.error('Failed to load demo questions:', error);
            this.elements.demoQuestions.innerHTML = '<div class="error">Failed to load demo questions</div>';
        }
    }

    renderDemoQuestions() {
        if (this.state.demoQuestions.length === 0) {
            this.elements.demoQuestions.innerHTML = '<div class="loading">No demo questions available</div>';
            return;
        }

        const questionsHTML = this.state.demoQuestions.map(q => 
            `<div class="demo-question" data-question="${q.question}">
                <i class="fas fa-question-circle"></i> ${q.question}
            </div>`
        ).join('');

        this.elements.demoQuestions.innerHTML = questionsHTML;

        // Add click handlers for demo questions
        this.elements.demoQuestions.querySelectorAll('.demo-question').forEach(el => {
            el.addEventListener('click', () => {
                const question = el.dataset.question;
                this.elements.chatInput.value = question;
                this.updateCharCount();
                this.updateSendButton();
                this.elements.chatInput.focus();
            });
        });
    }

    // File Upload
    async handleFileUpload(files) {
        if (this.state.isUploading) return;
        
        const pdfFiles = Array.from(files).filter(file => 
            file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf')
        );

        if (pdfFiles.length === 0) {
            this.showToast('Please select PDF files only', 'error');
            return;
        }

        for (const file of pdfFiles) {
            await this.uploadFile(file);
        }
    }

    async uploadFile(file) {
        if (file.size > 10 * 1024 * 1024) { // 10MB limit
            this.showToast(`File ${file.name} is too large (max 10MB)`, 'error');
            return;
        }

        this.state.isUploading = true;
        this.showUploadProgress(true);
        
        const formData = new FormData();
        formData.append('file', file);

        try {
            const xhr = new XMLHttpRequest();
            
            // Progress tracking
            xhr.upload.onprogress = (e) => {
                if (e.lengthComputable) {
                    const percent = Math.round((e.loaded / e.total) * 100);
                    this.updateUploadProgress(percent, `Uploading ${file.name}...`);
                }
            };

            // Response handling
            xhr.onload = () => {
                if (xhr.status === 200) {
                    const response = JSON.parse(xhr.responseText);
                    this.showToast(`Successfully uploaded: ${file.name}`, 'success');
                    this.loadDocuments(); // Refresh document list
                } else {
                    const error = JSON.parse(xhr.responseText);
                    this.showToast(`Upload failed: ${error.detail}`, 'error');
                }
                this.finishUpload();
            };

            xhr.onerror = () => {
                this.showToast(`Upload failed: ${file.name}`, 'error');
                this.finishUpload();
            };

            xhr.open('POST', '/upload');
            xhr.send(formData);

        } catch (error) {
            console.error('Upload error:', error);
            this.showToast(`Upload failed: ${error.message}`, 'error');
            this.finishUpload();
        }
    }

    showUploadProgress(show) {
        this.elements.uploadProgress.style.display = show ? 'block' : 'none';
    }

    updateUploadProgress(percent, text) {
        this.elements.progressFill.style.width = `${percent}%`;
        this.elements.progressText.textContent = text;
    }

    finishUpload() {
        this.state.isUploading = false;
        setTimeout(() => {
            this.showUploadProgress(false);
        }, 1000);
    }

    // Documents Management
    async loadDocuments() {
        try {
            const response = await fetch('/documents');
            const documents = await response.json();
            this.state.documents = documents;
            this.renderDocuments();
        } catch (error) {
            console.error('Failed to load documents:', error);
            this.elements.documentsList.innerHTML = '<div class="error">Failed to load documents</div>';
        }
    }

    renderDocuments() {
        if (this.state.documents.length === 0) {
            this.elements.documentsList.innerHTML = `
                <div class="no-documents">
                    <i class="fas fa-inbox"></i>
                    <p>No documents uploaded yet</p>
                </div>
            `;
            return;
        }

        const documentsHTML = this.state.documents.map(doc => `
            <div class="document-item" data-doc-id="${doc.id}">
                <div class="document-info">
                    <div class="document-name">
                        <i class="fas fa-file-pdf"></i> ${doc.filename}
                    </div>
                    <div class="document-meta">
                        ${doc.chunk_count} chunks
                    </div>
                </div>
                <div class="document-actions">
                    <button class="button danger" onclick="app.deleteDocument('${doc.id}', '${doc.filename}')">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </div>
        `).join('');

        this.elements.documentsList.innerHTML = documentsHTML;
    }

    async deleteDocument(docId, filename) {
        const confirmed = await this.showConfirmModal(
            `Are you sure you want to delete "${filename}"? This action cannot be undone.`
        );

        if (!confirmed) return;

        try {
            const response = await fetch(`/documents/${docId}`, {
                method: 'DELETE'
            });

            if (response.ok) {
                this.showToast(`Deleted: ${filename}`, 'success');
                this.loadDocuments();
            } else {
                const error = await response.json();
                this.showToast(`Delete failed: ${error.detail}`, 'error');
            }
        } catch (error) {
            console.error('Delete error:', error);
            this.showToast(`Delete failed: ${error.message}`, 'error');
        }
    }

    // Chat Functionality
    updateCharCount() {
        const count = this.elements.chatInput.value.length;
        this.elements.charCount.textContent = `${count}/500`;
        
        if (count > 450) {
            this.elements.charCount.style.color = 'var(--error-color)';
        } else if (count > 400) {
            this.elements.charCount.style.color = 'var(--warning-color)';
        } else {
            this.elements.charCount.style.color = 'var(--text-muted)';
        }
    }

    updateSendButton() {
        const hasText = this.elements.chatInput.value.trim().length > 0;
        const hasDocuments = this.state.documents.length > 0;
        this.elements.sendButton.disabled = !hasText || this.state.isProcessing;
        
        if (!hasDocuments && hasText) {
            this.elements.sendButton.title = 'Upload some PDF documents first';
        } else {
            this.elements.sendButton.title = '';
        }
    }

    async sendMessage() {
        const question = this.elements.chatInput.value.trim();
        if (!question || this.state.isProcessing) return;

        if (this.state.documents.length === 0) {
            this.showToast('Please upload some PDF documents first', 'warning');
            return;
        }

        // Add user message to chat
        this.addMessage('user', question);
        
        // Clear input
        this.elements.chatInput.value = '';
        this.updateCharCount();
        this.updateSendButton();

        // Show processing state
        this.state.isProcessing = true;
        this.showLoadingOverlay(true);
        
        // Add typing indicator
        const typingId = this.addMessage('bot', 'Thinking...', null, true);

        try {
            const response = await fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    question: question,
                    max_results: 5
                })
            });

            const data = await response.json();
            
            // Remove typing indicator
            this.removeMessage(typingId);

            if (response.ok) {
                // Add bot response
                this.addMessage('bot', data.answer, data.sources, false, data.confidence);
            } else {
                this.addMessage('bot', `Error: ${data.detail}`, null, false, 0);
                this.showToast('Failed to get answer', 'error');
            }

        } catch (error) {
            console.error('Question error:', error);
            this.removeMessage(typingId);
            this.addMessage('bot', 'Sorry, I encountered an error processing your question.', null, false, 0);
            this.showToast('Failed to get answer', 'error');
        } finally {
            this.state.isProcessing = false;
            this.showLoadingOverlay(false);
            this.updateSendButton();
        }
    }

    addMessage(sender, text, sources = null, isTyping = false, confidence = null) {
        const messageId = `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        const avatar = sender === 'user' ? 
            '<i class="fas fa-user"></i>' : 
            '<i class="fas fa-robot"></i>';

        let sourcesHTML = '';
        if (sources && sources.length > 0) {
            sourcesHTML = `
                <div class="sources">
                    <h4><i class="fas fa-link"></i> Sources:</h4>
                    ${sources.slice(0, 3).map(source => `
                        <div class="source-item">
                            <span>${source.source}</span>
                            <span class="confidence-score">${Math.round(source.similarity * 100)}%</span>
                        </div>
                    `).join('')}
                </div>
            `;
        }

        const messageHTML = `
            <div class="message ${sender}-message" id="${messageId}">
                <div class="message-avatar">${avatar}</div>
                <div class="message-content">
                    <p>${isTyping ? '🤔 ' + text : text}</p>
                    ${sourcesHTML}
                </div>
            </div>
        `;

        this.elements.chatMessages.insertAdjacentHTML('beforeend', messageHTML);
        this.scrollChatToBottom();
        
        return messageId;
    }

    removeMessage(messageId) {
        const messageEl = document.getElementById(messageId);
        if (messageEl) {
            messageEl.remove();
        }
    }

    scrollChatToBottom() {
        this.elements.chatMessages.scrollTop = this.elements.chatMessages.scrollHeight;
    }

    // UI Helpers
    showLoadingOverlay(show) {
        this.elements.loadingOverlay.style.display = show ? 'flex' : 'none';
    }

    showToast(message, type = 'info') {
        const toastId = `toast-${Date.now()}`;
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.id = toastId;
        toast.innerHTML = `
            <div style="display: flex; align-items: center; gap: 8px;">
                <i class="fas fa-${this.getToastIcon(type)}"></i>
                <span>${message}</span>
            </div>
        `;

        this.elements.toastContainer.appendChild(toast);

        // Auto remove after 4 seconds
        setTimeout(() => {
            const toastEl = document.getElementById(toastId);
            if (toastEl) {
                toastEl.style.animation = 'slideIn 0.3s ease reverse';
                setTimeout(() => toastEl.remove(), 300);
            }
        }, 4000);
    }

    getToastIcon(type) {
        const icons = {
            success: 'check-circle',
            error: 'exclamation-circle',
            warning: 'exclamation-triangle',
            info: 'info-circle'
        };
        return icons[type] || 'info-circle';
    }

    showConfirmModal(message) {
        return new Promise((resolve) => {
            this.elements.confirmMessage.textContent = message;
            this.elements.confirmModal.style.display = 'flex';

            const handleConfirm = () => {
                this.hideModal();
                resolve(true);
                cleanup();
            };

            const handleCancel = () => {
                this.hideModal();
                resolve(false);
                cleanup();
            };

            const cleanup = () => {
                this.elements.confirmOk.removeEventListener('click', handleConfirm);
                this.elements.confirmCancel.removeEventListener('click', handleCancel);
            };

            this.elements.confirmOk.addEventListener('click', handleConfirm);
            this.elements.confirmCancel.addEventListener('click', handleCancel);
        });
    }

    hideModal() {
        this.elements.confirmModal.style.display = 'none';
    }

    // Public methods for global access
    static getInstance() {
        if (!window.pdfQuestionAppInstance) {
            window.pdfQuestionAppInstance = new PDFQuestionApp();
        }
        return window.pdfQuestionAppInstance;
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = PDFQuestionApp.getInstance();
    
    // Global error handler
    window.addEventListener('error', (e) => {
        console.error('Global error:', e.error);
        if (window.app) {
            window.app.showToast('An unexpected error occurred', 'error');
        }
    });

    // Handle unhandled promise rejections
    window.addEventListener('unhandledrejection', (e) => {
        console.error('Unhandled promise rejection:', e.reason);
        if (window.app) {
            window.app.showToast('A network error occurred', 'error');
        }
    });
});

// Periodic health checks (every 30 seconds)
setInterval(() => {
    if (window.app) {
        window.app.checkSystemHealth();
    }
}, 30000);