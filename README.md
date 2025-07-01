# PDF Question Answering System

A powerful, cost-effective web application that allows users to upload PDF documents and ask questions about their content using AI. Built with FastAPI, ChromaDB, and support for multiple LLM backends including free local options.

## 🎯 Key Features

- **📄 PDF Processing**: Upload and extract text from PDF documents using pdfplumber
- **🤖 AI-Powered Q&A**: Ask questions in natural language and get accurate answers
- **💰 Cost-Optimized**: Multiple free options including local LLMs (Ollama)
- **🔍 Semantic Search**: ChromaDB vector database for relevant context retrieval
- **💻 Professional UI**: Clean, responsive web interface
- **🚀 Demo-Ready**: Portable, easy to demonstrate in any environment
- **⚡ Fast Setup**: One-click startup scripts for Windows and Linux/Mac

## 🏗️ Architecture

### Backend (FastAPI)
- **PDF Upload & Processing**: Text extraction with intelligent chunking
- **Vector Search**: ChromaDB with sentence-transformers embeddings
- **Multi-LLM Support**: Ollama (local), Groq (free), OpenAI (paid)
- **RESTful API**: Clean endpoints for all functionality

### Frontend (HTML/CSS/JavaScript)
- **Drag & Drop Upload**: Intuitive file upload interface
- **Real-time Chat**: Interactive question-answering interface
- **Document Management**: View and delete uploaded documents
- **Demo Mode**: Pre-loaded questions for easy demonstration

## 🚀 Quick Start

### Option 1: One-Click Demo (Recommended)

**Windows:**
```bash
# Double-click run_demo.bat or run in command prompt:
run_demo.bat
```

**Linux/Mac:**
```bash
# Make executable and run:
chmod +x run_demo.sh
./run_demo.sh
```

### Option 2: Manual Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Application**
   ```bash
   python main.py
   ```

3. **Access Interface**
   - Open browser to `http://localhost:8000`
   - Upload PDF documents
   - Start asking questions!

## 💰 Cost-Effective LLM Options

### 🆓 Completely Free Options

1. **Ollama (Recommended for Demos)**
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull a model (choose based on your hardware)
   ollama pull phi3:mini      # 2.3GB - Runs on 4GB RAM
   ollama pull llama3.1:8b    # 4.7GB - Runs on 8GB RAM
   ollama pull mistral:7b     # 4.1GB - Good balance
   ```

2. **Groq (700 requests/day free)**
   - Get free API key from [groq.com](https://groq.com)
   - Very fast inference (up to 500 tokens/second)
   - Set `GROQ_API_KEY` environment variable

### 💳 Paid Options (For Production)

3. **OpenAI GPT Models**
   - High accuracy and capability
   - Set `OPENAI_API_KEY` environment variable
   - $5 free credit for new users

## 📁 Project Structure

```
UserGuide/
├── main.py                    # FastAPI backend application
├── requirements.txt           # Python dependencies
├── run_demo.bat              # Windows startup script
├── run_demo.sh               # Linux/Mac startup script
├── .env.example              # Environment configuration template
├── README.md                 # This file
├── static/
│   ├── style.css            # Professional UI styling
│   └── script.js            # Frontend JavaScript
├── templates/
│   └── index.html           # Main web interface
├── uploads/                 # PDF file storage
├── vector_db/              # ChromaDB persistence
└── demo_data/
    ├── demo_questions.json  # Demo questions
    └── sample_user_guide.txt # Sample documentation
```

## 🔧 Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Free options
GROQ_API_KEY=your_groq_key_here
OLLAMA_HOST=http://localhost:11434

# Paid options
OPENAI_API_KEY=your_openai_key_here

# App settings
PORT=8000
DEBUG=false
```

### LLM Priority Order

The system automatically tries LLM backends in this order:
1. **Ollama** (local) - if running
2. **Groq** (free) - if API key provided
3. **OpenAI** (paid) - if API key provided

## 🎪 Demo Mode

Perfect for client presentations and office demonstrations:

### Pre-loaded Features
- **Sample Questions**: Click demo questions to test instantly
- **Professional UI**: Clean interface suitable for business demos
- **System Status**: Real-time health indicators
- **Portable**: Works offline with local LLMs

### Demo Script
1. **Show Upload**: Drag and drop a PDF document
2. **Ask Questions**: Use provided demo questions or custom ones
3. **Show Sources**: Highlight source attribution and confidence scores
4. **System Health**: Show multi-LLM backend support

## 📊 Performance & Accuracy

### Accuracy Optimizations
- **Smart Chunking**: Preserves paragraph boundaries (400-600 chars)
- **Hybrid Search**: Combines semantic similarity with keyword matching
- **Source Attribution**: Shows which documents informed each answer
- **Confidence Scoring**: Reliability indicators for responses

### Performance Tips
- **Local LLMs**: 2-5 second response times with Ollama
- **Cloud APIs**: Sub-second response with Groq
- **Memory Usage**: 4-8GB RAM recommended
- **Storage**: Efficient vector embeddings with ChromaDB

## 🛠️ Troubleshooting

### Common Issues

**"No LLM available"**
- Install Ollama: `curl -fsSL https://ollama.ai/install.sh | sh`
- Pull a model: `ollama pull phi3:mini`
- Or set up Groq/OpenAI API keys

**PDF extraction fails**
- Ensure PDF contains searchable text (not scanned images)
- Check file size is under 10MB
- Try a different PDF file

**Slow responses**
- Use smaller models (phi3:mini vs llama3.1:8b)
- Ensure adequate RAM
- Try local Ollama vs cloud APIs

## 🔒 Security & Privacy

### Data Protection
- **Local Storage**: All documents stored locally by default
- **No External Data**: Vector embeddings generated locally
- **API Key Security**: Environment variable configuration
- **Session Management**: Secure session handling

### Privacy Features
- **Offline Capable**: Works completely offline with Ollama
- **No Data Persistence**: Chat history not stored permanently
- **Local Processing**: Text analysis done locally

## 🚢 Deployment Options

### Local Development
```bash
python main.py
# Access at http://localhost:8000
```

### Production Deployment
```bash
# Using Gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 main:app

# Using Docker (create Dockerfile)
docker build -t pdf-qa .
docker run -p 8000:8000 pdf-qa
```

### Cloud Deployment
- **Railway**: Free tier with automatic deployments
- **Render**: Free tier with persistent storage
- **Heroku**: Easy deployment with add-ons

## 📈 Scaling Considerations

### For Larger Deployments
- **Database**: Replace ChromaDB with PostgreSQL + pgvector
- **File Storage**: Use S3 or cloud storage for PDFs
- **Caching**: Add Redis for response caching
- **Load Balancing**: Multiple backend instances
- **Monitoring**: Add logging and metrics

## 🤝 Contributing

### Development Setup
1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run in development mode: `python main.py`
4. Access at `http://localhost:8000`

### Code Structure
- **Backend**: FastAPI with async support
- **Frontend**: Vanilla JavaScript (no frameworks)
- **Database**: ChromaDB for vector storage
- **ML**: sentence-transformers for embeddings

## 📝 License

This project is open source. Feel free to use, modify, and distribute according to your needs.

## 🆘 Support

### Getting Help
1. Check the troubleshooting section above
2. Review system health status in the app
3. Ensure all dependencies are installed correctly
4. Test with provided demo questions first

### Feature Requests
- Enhanced PDF processing (OCR for scanned documents)
- Multi-language support
- Advanced search filters
- User authentication and multi-tenancy
- Document collaboration features

---

**Ready to start?** Run `./run_demo.sh` (Linux/Mac) or `run_demo.bat` (Windows) and visit `http://localhost:8000`!"# userguide" 
