"""
===============================================================
File: compare_latency.py
Author: SK Sahil
Project: Bachelor's Thesis – From Transformers to Agents
Experiment: Latency and Token Usage Comparison
===============================================================
Description:
Compares the inference performance of TinyLlama and Phi-2 models
using data from token_usage.csv and latency_results.json.

Generates:
 - A JSON summary of comparative latency and efficiency
 - Console summary table for visual verification

Results saved to:
  /results/output_logs/latency_results.json
===============================================================
"""

import json
import pandas as pd
from pathlib import Path
from tabulate import tabulate

# === Define project base directory ===
BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "results/output_logs"
CSV_PATH = RESULTS_DIR / "token_usage.csv"
JSON_PATH = RESULTS_DIR / "latency_results.json"

# === Ensure paths exist ===
if not CSV_PATH.exists():
    raise FileNotFoundError(f"❌ Missing file: {CSV_PATH}")
if not JSON_PATH.exists():
    raise FileNotFoundError(f"❌ Missing file: {JSON_PATH}")

print(f"📂 Loaded paths:\n- CSV: {CSV_PATH}\n- JSON: {JSON_PATH}\n")

# === Read CSV for token & latency data ===
df = pd.read_csv(CSV_PATH)

# === Display token usage comparison ===
print("📊 Token and Latency Comparison (from CSV)\n")
print(tabulate(df, headers="keys", tablefmt="github", showindex=False))

# === Read JSON latency log ===
with open(JSON_PATH, "r", encoding="utf-8") as f:
    latency_data = json.load(f)

# === Summarize latency by model ===
summary = {}
for entry in latency_data:
    model = entry.get("model_name")
    latency = entry.get("runtime_seconds")
    summary[model] = {
        "latency_seconds": latency,
        "environment": entry.get("system"),
        "python_version": entry.get("python_version")
    }

# === Print summary ===
print("\n====================== SUMMARY REPORT ======================")
for model, info in summary.items():
    print(f"Model: {model}")
    print(f"  → Latency: {info['latency_seconds']} s")
    print(f"  → System: {info['environment']}")
    print(f"  → Python: {info['python_version']}\n")
print("=============================================================\n")

# === Save summarized comparison JSON ===
OUTPUT_JSON = RESULTS_DIR / "latency_comparison_summary.json"
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)

print(f"✅ Comparison summary saved to: {OUTPUT_JSON}")
print("🎯 Task Completed: You can now include this data in your appendix PDF.")
