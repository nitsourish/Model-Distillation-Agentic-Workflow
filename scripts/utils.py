"""
Utility functions for loan prediction application.
"""

import json
import numpy as np
import os
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(env_path)


def _make_json_serializable(obj):
    """Convert objects to JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def setup_azure_openai_model():
    """Setup Azure OpenAI model with credentials."""
    if not os.getenv("AZURE_OPENAI_API_KEY") or not os.getenv("AZURE_OPENAI_ENDPOINT"):
        raise ConnectionError(
            "Azure OpenAI credentials not found. Please ensure AZURE_OPENAI_API_KEY and "
            "AZURE_OPENAI_ENDPOINT are set in your .env file or environment variables."
        )
    
    return AzureChatOpenAI(
        azure_deployment="gpt-4.1-nano",
        api_version="2024-12-01-preview",
        model="gpt-4.1-nano"
    )


def format_probability_change(new_prob, baseline_prob):
    """Format probability change with arrow and percentage."""
    prob_change = new_prob - baseline_prob
    arrow = "↗️" if prob_change > 0 else "↘️" if prob_change < 0 else "➡️"
    return f"{arrow} {prob_change:+.1%}"
