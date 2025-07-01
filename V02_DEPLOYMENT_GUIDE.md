# InteleOrchestrator Assistant (v0.2) - Deployment Guide

## Overview

Version 0.2 is a domain-specific, pre-configured Q&A application designed specifically for InteleOrchestrator system support. Unlike v0.1, this version requires zero user setup - documents are pre-loaded and the model is pre-configured.

## Key Features

### 🎯 Domain-Specific Focus
- **Pre-configured for InteleOrchestrator system support**
- Auto-loads 7 InteleOrchestrator documents on startup
- Specialized system prompts for medical workflow context
- No file upload or model selection needed

### 🚀 Streamlined User Experience
- Clean, focused chat interface
- Instant answers (after initial knowledge base loading)
- Progress tracking during document processing
- Source citations from InteleOrchestrator documentation

### ⚙️ Configuration-Driven
- Single `app_config.json` controls all settings
- Easy to modify for different domains
- Pre-configured for best free model (Groq Llama 3 8B)

## Documents Included

The knowledge base contains:
1. **Clario Smart Worklist PACS Admin Guide**
2. **IAVIC InteleOrchestrator Admin Training Oceania**
3. **InteleOrchestrator Radiologist Training Checklist**
4. **InteleOrchestrator System Requirements Guide**
5. **InteleOrchestrator 4.5 User Guide**
6. **Peer Review Training**
7. **Unofficial InteleOrchestrator User Role Permissions Admin Guide**

## Deployment Steps

### 1. Repository Setup
```bash
# Switch to v0.2 branch
git checkout v0.2-domain-specific

# Push to GitHub
git push origin v0.2-domain-specific
```

### 2. Streamlit Cloud Deployment
1. Go to https://share.streamlit.io/
2. Click "New app"
3. Select your repository
4. **Important Settings:**
   - **Branch:** `v0.2-domain-specific`
   - **Main file:** `streamlit_app_v02.py`
   - **App URL:** Choose something like `inteleorchestrator-assistant`

### 3. API Key Configuration
In Streamlit Cloud app settings, add:
```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

**Get Groq API Key (Free):**
1. Visit https://console.groq.com/
2. Sign up for free account
3. Create API key
4. Add to Streamlit secrets

## User Experience Flow

### First Visit
1. User visits app URL
2. App shows "🔄 Loading InteleOrchestrator knowledge base..."
3. Progress bar shows document processing (30-60 seconds)
4. Clean chat interface appears

### Subsequent Visits
1. Knowledge base loads from cache (instant)
2. User immediately sees chat interface
3. Can ask questions right away

## Sample Questions to Test

- "How do I set up user permissions in InteleOrchestrator?"
- "What are the system requirements for InteleOrchestrator 4.5?"
- "How do I configure the smart worklist?"
- "What training is required for radiologists?"
- "How do I troubleshoot peer review issues?"

## Configuration Customization

To modify for different domains, edit `app_config.json`:

```json
{
  "app_info": {
    "title": "Your System Assistant",
    "description": "Get instant answers about your system",
    "icon": "🏥",
    "domain": "your_domain"
  },
  "system_prompt": "Your specialized system prompt here..."
}
```

## Advantages Over v0.1

| Feature | v0.1 (General) | v0.2 (Domain-Specific) |
|---------|----------------|-------------------------|
| Setup Required | Upload docs, select model | None - ready to use |
| First Use | Manual document upload | Auto-loads knowledge base |
| UI Complexity | Multiple controls | Single chat interface |
| Domain Focus | Generic Q&A | InteleOrchestrator specific |
| User Training | Requires explanation | Intuitive single-purpose |

## Deployment URL

Once deployed, your app will be available at:
`https://your-app-name.streamlit.app`

Perfect for sharing with medical staff who need quick InteleOrchestrator support without any technical setup.