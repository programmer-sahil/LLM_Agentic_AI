"""
===============================================================
File: inference_phi2.py
Author: SK Sahil
Project: Bachelor's Thesis – From Transformers to Agents
Experiment: Phi-2 Inference and Comparison
===============================================================
Description:
Runs a local inference test using the Phi-2 model to evaluate:
- Text generation latency and token usage
- Output quality for academic reasoning prompts
- Comparison with TinyLlama baseline (stored in shared CSV/JSON)

Results are automatically saved to:
  /results/generated_text_samples/phi2_output.txt
  /results/output_logs/token_usage.csv
  /results/output_logs/latency_results.json
===============================================================
"""

# === Imports ===
import time
import json
import csv
import platform
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# === Define project base path ===
BASE_DIR = Path(__file__).resolve().parents[2]

# === Define result paths ===
RESULTS_TEXT = BASE_DIR / "results/generated_text_samples/phi2_output.txt"
RESULTS_CSV = BASE_DIR / "results/output_logs/token_usage.csv"
RESULTS_JSON = BASE_DIR / "results/output_logs/latency_results.json"

# === Ensure folders exist ===
RESULTS_TEXT.parent.mkdir(parents=True, exist_ok=True)
RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)

# === Model Info ===
MODEL_ID = "microsoft/phi-2"  # ~2.7B parameters
print(f"\n🔹 Loading model: {MODEL_ID} ...")

start_load = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
load_time = time.time() - start_load

print(f"✅ Model loaded successfully in {load_time:.2f} seconds.\n")

# === Prompt for inference ===
prompt = (
    "Explain in 5 clear bullet points how attention mechanisms enable "
    "transformer models to understand long-range dependencies in text."
)

# === Generate Response ===
print("🧠 Generating output using Phi-2 model...")
start = time.time()
response = generator(prompt, max_new_tokens=180, do_sample=False)[0]["generated_text"]
end = time.time()
latency = end - start

# === Token Statistics ===
input_tokens = len(tokenizer(prompt)["input_ids"])
output_tokens = len(tokenizer(response)["input_ids"]) - input_tokens

# === Print Summary ===
print("\n====================== SUMMARY ======================")
print(f"Model: {MODEL_ID}")
print(f"Prompt Tokens: {input_tokens}")
print(f"Generated Tokens: {output_tokens}")
print(f"Latency: {latency:.2f} seconds")
print("=====================================================\n")
print(f"Generated Text:\n{response}\n")

# === Save Outputs ===
print("💾 Saving results...")

# Save generated text
with open(RESULTS_TEXT, "w", encoding="utf-8") as f:
    f.write(response)

# Append token & latency stats to existing CSV (created by TinyLlama script)
write_header = not RESULTS_CSV.exists()
with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    if write_header:
        writer.writerow(["model", "input_tokens", "output_tokens", "latency_sec"])
    writer.writerow(["Phi-2", input_tokens, output_tokens, round(latency, 2)])

# Update latency comparison JSON
latency_data = {
    "model_name": MODEL_ID,
    "runtime_seconds": round(latency, 2),
    "system": platform.platform(),
    "python_version": platform.python_version(),
    "note": "Phi-2 inference test for LLM evolution phase comparison"
}

if RESULTS_JSON.exists():
    try:
        with open(RESULTS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        data = []
else:
    data = []

data.append(latency_data)
with open(RESULTS_JSON, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)

print("✅ All results saved successfully!")
print("📂 Files generated:")
print(f"   - {RESULTS_TEXT}")
print(f"   - {RESULTS_CSV}")
print(f"   - {RESULTS_JSON}")
print("\n=====================================================")
print("🎯 Task Completed: You can now move to model comparison visualization.")
