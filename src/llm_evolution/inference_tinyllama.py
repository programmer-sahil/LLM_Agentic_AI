"""
===============================================================
File: inference_tinyllama.py
Author: SK Sahil
Project: Bachelor's Thesis – From Transformers to Agents
Experiment: TinyLlama-1.1B Inference Test
===============================================================
Description:
Runs a local inference test using the TinyLlama-1.1B model to evaluate:
- Text generation latency
- Input/output token usage
- Output quality for baseline comparison

Output is saved automatically to:
  /results/generated_text_samples/tinyllama_output.txt
  /results/output_logs/token_usage.csv
  /results/output_logs/inference_meta.json
===============================================================
"""

# === Imports ===
import time
import json
import csv
import platform
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# === Define project base path (auto-detect root) ===
# This automatically points two levels up from this file
BASE_DIR = Path(__file__).resolve().parents[2]

# === Define result paths ===
RESULTS_TEXT = BASE_DIR / "results/generated_text_samples/tinyllama_output.txt"
RESULTS_CSV = BASE_DIR / "results/output_logs/token_usage.csv"
RESULTS_META = BASE_DIR / "results/output_logs/inference_meta.json"

# === Ensure folders exist ===
RESULTS_TEXT.parent.mkdir(parents=True, exist_ok=True)
RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)

# === Model info ===
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"\n Loading model: {MODEL_ID} ...")

start_load = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
load_time = time.time() - start_load

print(f" Model loaded successfully in {load_time:.2f} seconds.\n")

# === Prompt for inference ===
prompt = (
    "In 5 concise bullet points, explain what 'Retrieval-Augmented Generation (RAG)' is "
    "and how it improves the capabilities of Large Language Models."
)

# === Generate response ===
print(" Generating output...")
start_time = time.time()
response = generator(prompt, max_new_tokens=150, do_sample=False)[0]["generated_text"]
end_time = time.time()
latency = end_time - start_time

# === Token statistics ===
input_tokens = len(tokenizer(prompt)["input_ids"])
output_tokens = len(tokenizer(response)["input_ids"]) - input_tokens

# === Console summary ===
print("\n====================== SUMMARY ======================")
print(f"Model: {MODEL_ID}")
print(f"Prompt Tokens: {input_tokens}")
print(f"Generated Tokens: {output_tokens}")
print(f"Latency: {latency:.2f} seconds")
print("=====================================================\n")
print(f"Generated Text:\n{response}\n")

# === Save outputs ===
print(" Saving results...")

# Save generated text
with open(RESULTS_TEXT, "w", encoding="utf-8") as f:
    f.write(response)

# Save token/latency stats to CSV
with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["model", "input_tokens", "output_tokens", "latency_sec"])
    writer.writerow(["TinyLlama-1.1B", input_tokens, output_tokens, round(latency, 2)])

# Save metadata JSON
meta = {
    "model_name": MODEL_ID,
    "runtime_seconds": latency,
    "system": platform.platform(),
    "python_version": platform.python_version(),
    "total_tokens": input_tokens + output_tokens,
    "device": "auto (CPU or MPS)",
    "note": "TinyLlama inference test for B.Sc. thesis (LLM evolution phase)"
}
with open(RESULTS_META, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=4)

print(" All results saved successfully!")
print(" Files generated:")
print(f"   - {RESULTS_TEXT}")
print(f"   - {RESULTS_CSV}")
print(f"   - {RESULTS_META}")
print("\n=====================================================")
print(" Task Completed: You can now move to Phi-2 inference next.")
