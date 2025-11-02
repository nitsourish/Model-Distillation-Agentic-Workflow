# 🏦 Privacy-Aware Loan Agent - Scripts

This directory contains the modular Streamlit application for the Privacy-Aware Loan Approval Agent.

## 📁 File Structure

- **`app.py`**: Main Streamlit application with UI
- **`agent.py`**: Agent classes using LangGraph 
- **`tools.py`**: Tool definitions for data and predictions
- **`utils.py`**: Common utility functions
- **`prediction_utils.py`**: ML model utilities

## 🚀 Quick Start

1. **From the main directory**, run:
   ```bash
   ./start_app.sh
   ```

2. **Or manually from scripts directory**:
   ```bash
   cd scripts
   streamlit run app.py
   ```

3. **Open browser**: http://localhost:8501

## 🔧 Module Dependencies

```
app.py
├── utils.py (Azure OpenAI setup)
├── tools.py (Data & prediction tools)
│   ├── utils.py (JSON serialization)
│   └── prediction_utils.py (ML models)
└── agent.py (LangGraph agents)
```

## ✅ Verification

All modules have been tested and verified:
- ✅ Syntax checking passed
- ✅ Import dependencies resolved  
- ✅ Basic tool functionality working
- ✅ Streamlit compatibility confirmed

## 🎯 Features

- **Privacy Protection**: Context isolation between customers
- **What-if Analysis**: LLM-powered scenario modifications
- **Real-time Predictions**: ML models for loan approval
- **Interactive UI**: Streamlit interface with metrics
- **Cost Optimized**: Efficient LLM usage patterns

# Azure OpenAI Setup

To use the privacy-preserving loan approval agent, you need to set up your Azure OpenAI credentials:

1. **Create/Open a `.env` file in the `model_distillation/` directory.**
2. **Add your Azure OpenAI API key and endpoint:**

```
AZURE_OPENAI_API_KEY=<your_api_key>
AZURE_OPENAI_ENDPOINT=<your_endpoint>
```

- Replace `<your_api_key>` with your Azure OpenAI API key.
- Replace `<your_endpoint>` with your Azure OpenAI endpoint URL (e.g., `https://<your-resource-name>.openai.azure.com/`).

**Note:**
- The application will automatically load these values from the `.env` file.
- Never share your API key publicly.
- If you change your credentials, update the `.env` file and restart the app.

---

3. **Restart the app to load the environment variables.**

