"""
====================================================================
File: visualize_latency.py
Author: SK Sahil
Project: Bachelor's Thesis – From Transformers to Agents
Experiment: Visualization of Latency & Token Usage Comparison
====================================================================

Description:
Generates visual comparisons between TinyLlama-1.1B and Phi-2 
using CSV and JSON logs created from previous experiments.

Outputs:
- latency_bar_chart.png
- token_usage_chart.png
- latency_vs_tokens_scatter.png

All figures are saved inside:
  /results/visualizations/

====================================================================
"""

# === Imports ===
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# === Configure plot style ===
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.edgecolor": "#dee2e6",
    "axes.labelcolor": "#212529",
    "xtick.color": "#212529",
    "ytick.color": "#212529",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "bold"
})

# === Define base directories ===
BASE_DIR = Path(__file__).resolve().parents[2]
CSV_PATH = BASE_DIR / "results/output_logs/token_usage.csv"
JSON_PATH = BASE_DIR / "results/output_logs/latency_comparison_summary.json"
VIS_DIR = BASE_DIR / "results/visualizations"
VIS_DIR.mkdir(parents=True, exist_ok=True)

print(" Loaded paths:")
print(f"- CSV: {CSV_PATH}")
print(f"- JSON: {JSON_PATH}\n")

# === Load CSV and JSON Data ===
try:
    df = pd.read_csv(CSV_PATH)
    with open(JSON_PATH, "r") as f:
        latency_summary = json.load(f)
except Exception as e:
    print(f" Error loading data: {e}")
    exit()

print(" Data loaded successfully!\n")

# === Plot 1: Latency Comparison ===
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(df["model"], df["latency_sec"], color=["#5e60ce", "#64dfdf"], edgecolor="black")

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.2f}s", ha='center', fontsize=11)

ax.set_title("Model Latency Comparison", pad=15)
ax.set_xlabel("Model")
ax.set_ylabel("Latency (seconds)")
ax.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(VIS_DIR / "latency_bar_chart.png", dpi=300)
plt.close()

# === Plot 2: Token Usage Comparison ===
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(df["model"], df["input_tokens"], color="#48cae4", label="Input Tokens", width=0.4)
ax.bar(df["model"], df["output_tokens"], bottom=df["input_tokens"], color="#0077b6", label="Output Tokens", width=0.4)
ax.set_title("Token Usage per Model", pad=15)
ax.set_xlabel("Model")
ax.set_ylabel("Number of Tokens")
ax.legend()
plt.tight_layout()
plt.savefig(VIS_DIR / "token_usage_chart.png", dpi=300)
plt.close()

# === Plot 3: Latency vs. Total Tokens ===
df["total_tokens"] = df["input_tokens"] + df["output_tokens"]
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.scatter(df["total_tokens"], df["latency_sec"], s=150, color="#ef476f", alpha=0.8, edgecolors="black")
for i, row in df.iterrows():
    ax.text(row["total_tokens"] + 5, row["latency_sec"], row["model"], fontsize=11, weight="bold")

ax.set_title("Latency vs Total Tokens")
ax.set_xlabel("Total Tokens (input + output)")
ax.set_ylabel("Latency (seconds)")
ax.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(VIS_DIR / "latency_vs_tokens_scatter.png", dpi=300)
plt.close()

# === Summary ===
print(" Visualization Completed Successfully!\n")
print(" Figures saved to:")
print(f"   - {VIS_DIR / 'latency_bar_chart.png'}")
print(f"   - {VIS_DIR / 'token_usage_chart.png'}")
print(f"   - {VIS_DIR / 'latency_vs_tokens_scatter.png'}\n")
print(" Task Completed: Figures ready for inclusion in thesis appendix.")
