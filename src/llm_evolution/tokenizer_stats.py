"""
====================================================================
File: tokenizer_stats.py
Author: SK Sahil
Project: Bachelor's Thesis – From Transformers to Agents
Experiment: Tokenizer Statistics and Vocabulary Efficiency
====================================================================

Description:
This script analyzes the tokenizer behavior of both TinyLlama and Phi-2
models, measuring vocabulary size, average tokens per sentence, and
compression ratio for text inputs.

Outputs:
- tokenizer_comparison.json
- tokenizer_summary.txt

All results saved to:
  /results/output_logs/
====================================================================
"""

# === Imports ===
from transformers import AutoTokenizer
import json
import statistics
from pathlib import Path

# === Define Models ===
MODELS = {
    "TinyLlama-1.1B": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Phi-2": "microsoft/phi-2"
}

# === Test Sentences for Tokenization ===
TEST_TEXTS = [
    "Artificial intelligence enables machines to perform human-like reasoning.",
    "Transformers revolutionized NLP by introducing attention mechanisms.",
    "Large Language Models are fine-tuned for domain-specific applications.",
    "Phi-2 model demonstrates efficient token usage compared to TinyLlama.",
    "Tokenization helps convert human language into numerical input for models."
]

# === Define Paths ===
BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "results" / "output_logs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === Function to Analyze Tokenizer ===
def analyze_tokenizer(model_name, model_id):
    print(f"\n🔍 Analyzing Tokenizer for {model_name} ({model_id})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    vocab_size = len(tokenizer)
    token_counts = [len(tokenizer.encode(text)) for text in TEST_TEXTS]
    avg_tokens = round(statistics.mean(token_counts), 2)
    compression_ratio = round(sum(len(text.split()) for text in TEST_TEXTS) / sum(token_counts), 2)

    print(f"✅ Vocabulary Size: {vocab_size}")
    print(f"🧮 Avg Tokens per Sentence: {avg_tokens}")
    print(f"📊 Compression Ratio: {compression_ratio}\n")

    return {
        "model": model_name,
        "vocab_size": vocab_size,
        "avg_tokens_per_sentence": avg_tokens,
        "compression_ratio": compression_ratio,
        "sample_token_counts": token_counts
    }

# === Main Execution ===
if __name__ == "__main__":
    print("📘 Starting Tokenizer Statistics Analysis...\n")

    results = []
    for model_name, model_id in MODELS.items():
        stats = analyze_tokenizer(model_name, model_id)
        results.append(stats)

    # === Save JSON Output ===
    json_path = OUTPUT_DIR / "tokenizer_comparison.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    # === Save Text Summary ===
    summary_path = OUTPUT_DIR / "tokenizer_summary.txt"
    with open(summary_path, "w") as f:
        for r in results:
            f.write(f"Model: {r['model']}\n")
            f.write(f"Vocab Size: {r['vocab_size']}\n")
            f.write(f"Avg Tokens/Sentence: {r['avg_tokens_per_sentence']}\n")
            f.write(f"Compression Ratio: {r['compression_ratio']}\n")
            f.write(f"Token Counts: {r['sample_token_counts']}\n")
            f.write("="*50 + "\n")

    print("💾 Results Saved Successfully!")
    print(f"📂 JSON: {json_path}")
    print(f"📂 Summary: {summary_path}")
    print("\n🎯 Task Completed: Tokenizer statistics ready for visualization or appendix inclusion.")
