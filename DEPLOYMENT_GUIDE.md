# PDF Question-Answering App - Enhanced Version

## Overview

The enhanced PDF Q&A application (`streamlit_app_enhanced.py`) includes advanced features like multiple LLM providers, progress tracking, and model selection UI.

## Key Features

### 🤖 Multiple LLM Providers
- **Groq**: Fast, free models (Llama 3 8B/70B, Mixtral 8x7B)
- **OpenRouter**: Access to cutting-edge models including DeepSeek V3
- **OpenAI**: GPT-3.5 Turbo and GPT-4o Mini
- Real-time provider status and model switching

### 📊 Enhanced UI
- Model selection dropdown with quality/speed indicators
- Real-time API status monitoring
- Processing progress with time estimation
- Response metadata (model, timing, tokens)
- Usage statistics tracking

### ⚡ Performance Features
- TF-IDF vector search (no ChromaDB dependency issues)
- Chunked text processing with overlap
- Progress tracking for PDF processing
- Async-ready architecture

## Deployment to Streamlit Cloud

### 1. Prepare Repository
```bash
# Your files should include:
streamlit_app_enhanced.py  # Main enhanced app
streamlit_app.py          # Simple fallback version
requirements.txt          # Dependencies
```

### 2. Set Up API Keys in Streamlit Secrets
In your Streamlit Cloud app settings, add these secrets:

```toml
# .streamlit/secrets.toml
GROQ_API_KEY = "your_groq_api_key_here"
OPENAI_API_KEY = "your_openai_api_key_here"  
OPENROUTER_API_KEY = "your_openrouter_api_key_here"
```

### 3. API Key Setup

#### Groq (Free, Fast)
1. Visit https://console.groq.com/
2. Sign up for free account
3. Create API key
4. Add to Streamlit secrets as `GROQ_API_KEY`

#### OpenRouter (Free Tier Available)
1. Visit https://openrouter.ai/
2. Sign up for account
3. Get free credits ($1-5 typically)
4. Create API key
5. Add to Streamlit secrets as `OPENROUTER_API_KEY`

#### OpenAI (Paid)
1. Visit https://platform.openai.com/
2. Create account and add payment method
3. Create API key
4. Add to Streamlit secrets as `OPENAI_API_KEY`

### 4. Deploy
1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Select `streamlit_app_enhanced.py` as main file
4. Deploy!

## Available Models

### Free Models (Recommended for Demo)
- **Llama 3 8B** (Groq) - Fast, reliable
- **Llama 3 70B** (Groq) - More capable
- **Mixtral 8x7B** (Groq) - Mixture of experts
- **DeepSeek V3** (OpenRouter) - Latest, very capable
- **Llama 3.1 405B** (OpenRouter) - Massive model
- **Qwen 2.5 72B** (OpenRouter) - Alibaba's model

### Paid Models
- **GPT-3.5 Turbo** (OpenAI) - Fast, efficient
- **GPT-4o Mini** (OpenAI) - Latest efficient model

## Usage Instructions

1. **Upload PDFs**: Use sidebar file uploader
2. **Select Model**: Choose from dropdown based on your needs
3. **Ask Questions**: Use chat interface or demo questions
4. **Monitor Progress**: Watch real-time processing updates
5. **View Sources**: Expand source sections in responses

## Troubleshooting

### No Models Available
- Check API keys in Streamlit secrets
- Ensure keys are valid and have credits
- At minimum, set up Groq (free) for basic functionality

### Processing Errors
- Ensure PDFs contain extractable text
- Check file size limits (Streamlit Cloud: 200MB)
- Try smaller files first

### Slow Responses
- Use faster models (Llama 3 8B, GPT-3.5)
- Reduce chunk size in search
- Check API rate limits

## Performance Comparison

| Model | Provider | Speed | Quality | Cost | Best For |
|-------|----------|-------|---------|------|----------|
| Llama 3 8B | Groq | ⚡⚡⚡ | ⭐⭐ | 🆓 | Quick demos |
| Llama 3 70B | Groq | ⚡⚡ | ⭐⭐⭐ | 🆓 | Balanced |
| DeepSeek V3 | OpenRouter | ⚡⚡ | ⭐⭐⭐⭐ | 🆓 | Best free |
| GPT-4o Mini | OpenAI | ⚡⚡ | ⭐⭐⭐⭐ | 💰 | Production |

## Demo URL
Once deployed, share your Streamlit Cloud URL for instant access without local installation requirements.