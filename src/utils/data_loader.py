"""
===============================================================
File: data_loader.py
Author: SK Sahil
Project: Bachelor's Thesis – From Transformers to Agents
Module: Utility - Data Loading
===============================================================
Description:
Provides helper functions to load and preprocess evaluation text
samples for LLM inference and benchmarking.
===============================================================
"""

import json
from pathlib import Path


def load_sample_prompts():
    """
    Loads a set of predefined sample prompts used for inference.
    Returns:
        list[str]: A list of textual prompts.
    """
    return [
        "Explain the role of attention mechanism in transformer models.",
        "Compare the architecture of TinyLlama and Phi-2 models.",
        "Describe how tokenization affects inference performance.",
        "Summarize the key challenges in scaling agentic AI systems.",
    ]


def load_json_data(file_path):
    """
    Safely loads JSON data from the specified file.
    Args:
        file_path (str): Path to JSON file.
    Returns:
        dict | list: Parsed JSON content.
    """
    path = Path(file_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"File not found: {file_path}")


if __name__ == "__main__":
    print(" Sample Prompts Loaded:")
    for p in load_sample_prompts():
        print("-", p)
