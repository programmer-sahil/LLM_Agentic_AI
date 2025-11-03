"""
====================================================================
File: autogpt_demo.py
Author: SK Sahil
Project: Bachelor's Thesis – From Transformers to Agents
Experiment: AutoGPT-Style Self-Reflective Reasoning Agent
====================================================================
Description:
This script demonstrates how an LLM (Phi-2 or TinyLlama) can act as
an *autonomous self-reflective agent* — reasoning, generating an
answer, then reflecting to refine it (Think → Act → Reflect Loop).

Outputs:
  • Step-wise reasoning logs
  • Refined final output
  • Saved results for inclusion in the thesis appendix

Saved files:
  /results/agentic_logs/autogpt_reflection_log.txt
  /results/agentic_logs/autogpt_session_summary.json
====================================================================
"""

# === Imports ===
import os
import json
import time
from pathlib import Path
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


# === Setup project structure ===
BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "results" / "agentic_logs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = RESULTS_DIR / "autogpt_reflection_log.txt"
SUMMARY_JSON = RESULTS_DIR / "autogpt_session_summary.json"

# === Configuration ===
MODEL_ID = "microsoft/phi-2"       # lightweight model for reasoning
DEVICE = "cuda"                    # or "cpu" if Colab has no GPU
REFLECTION_CYCLES = 2              # number of think-reflect loops


# === Load model and tokenizer ===
print(f"\n🔹 Loading model: {MODEL_ID} ...")
start_load = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")
agent = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
load_time = time.time() - start_load
print(f"✅ Model loaded successfully in {load_time:.2f}s\n")


# === Agentic Reasoning Loop ===
def autogpt_reasoning(task_prompt: str, reflection_cycles: int = 2):
    """
    Performs iterative reasoning and reflection using a small language model.
    """
    logs = []
    current_prompt = task_prompt
    print("🧠 Starting AutoGPT-style reasoning...\n")

    for cycle in range(reflection_cycles):
        print(f"🔁 Cycle {cycle + 1}/{reflection_cycles}")
        start = time.time()

        response = agent(
            current_prompt,
            max_new_tokens=220,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )[0]["generated_text"]

        end = time.time()
        latency = round(end - start, 2)

        # Reflection phase
        reflection_prompt = (
            f"Reflect on this answer critically and suggest one improvement:\n{response}"
        )
        reflection = agent(
            reflection_prompt,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )[0]["generated_text"]

        log_entry = {
            "cycle": cycle + 1,
            "prompt": current_prompt,
            "response": response,
            "reflection": reflection,
            "latency_seconds": latency,
        }
        logs.append(log_entry)

        # Update the prompt for next cycle
        current_prompt = (
            f"Improve this answer based on the reflection:\n{reflection}\n\nPrevious answer:\n{response}"
        )

        print(f"✅ Cycle {cycle + 1} complete ({latency}s)\n")

    final_output = logs[-1]["response"] if logs else "No output"
    return final_output, logs


# === Run Experiment ===
task = (
    "You are an academic research assistant. "
    "Generate a concise 5-point summary explaining how autonomous agents "
    "extend the capabilities of transformer models in practical reasoning tasks."
)

final_answer, reasoning_logs = autogpt_reasoning(task, REFLECTION_CYCLES)

# === Save results ===
print("💾 Saving session logs...")
with open(LOG_FILE, "w", encoding="utf-8") as f:
    for entry in reasoning_logs:
        f.write(f"--- Cycle {entry['cycle']} ---\n")
        f.write(f"Prompt:\n{entry['prompt']}\n\n")
        f.write(f"Response:\n{entry['response']}\n\n")
        f.write(f"Reflection:\n{entry['reflection']}\n\n")
        f.write(f"Latency: {entry['latency_seconds']}s\n\n")

summary = {
    "model": MODEL_ID,
    "reflection_cycles": REFLECTION_CYCLES,
    "avg_latency": round(sum(e["latency_seconds"] for e in reasoning_logs) / len(reasoning_logs), 2),
    "final_output_excerpt": final_answer[:300],
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
}

with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)

print("✅ Session saved successfully!")
print(f"📂 Log File: {LOG_FILE}")
print(f"📂 Summary JSON: {SUMMARY_JSON}\n")
print("=======================================================")
print("🎯 Task Completed: AutoGPT Reflection Demo executed successfully.")
print("=======================================================\n")
