"""
====================================================================
File: open_devin_test.py
Author: SK Sahil
Project: Bachelor's Thesis – From Transformers to Agents
Experiment: Autonomous Developer Simulation (OpenDevin-Style)
====================================================================
Description:
Simulates an autonomous code-generation agent workflow inspired by
OpenDevin and CodeAgent. The agent receives a coding task, plans the
solution, generates Python code, and performs a mock review phase.

Outputs:
  • Developer reasoning log
  • Generated code snippet
  • JSON summary with timing and verdict

Saved files:
  /results/agentic_logs/open_devin_reasoning_log.txt
  /results/agentic_logs/open_devin_session_summary.json
====================================================================
"""

# === Imports ===
import json
import time
from pathlib import Path
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


# === Setup directories ===
BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "results" / "agentic_logs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = RESULTS_DIR / "open_devin_reasoning_log.txt"
SUMMARY_FILE = RESULTS_DIR / "open_devin_session_summary.json"


# === Configuration ===
MODEL_ID = "microsoft/phi-2"
print(f"\n🔹 Loading model: {MODEL_ID} ...")

start_load = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")
llm = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
load_time = round(time.time() - start_load, 2)

print(f"✅ Model loaded successfully in {load_time} seconds.\n")


# === Define the Developer Agent ===
def autonomous_dev_agent(task: str):
    """
    Simulates a multi-step autonomous coding agent.
    Steps:
      1. Understand problem and plan
      2. Generate code solution
      3. Review code for potential errors
    """
    reasoning_log = []
    start_time = time.time()

    # Step 1: Problem understanding and planning
    step1_prompt = (
        f"You are an autonomous coding agent. Analyze this programming task carefully:\n"
        f"{task}\n\n"
        f"Describe your step-by-step reasoning and plan the code structure clearly."
    )
    step1_response = llm(step1_prompt, max_new_tokens=180, do_sample=False)[0]["generated_text"]
    reasoning_log.append({"step": "problem_analysis", "response": step1_response})

    # Step 2: Code generation
    step2_prompt = (
        f"Now based on your previous reasoning, generate clean, readable Python code "
        f"that fulfills the task:\nTask: {task}\nInclude comments for clarity."
    )
    step2_response = llm(step2_prompt, max_new_tokens=220, do_sample=False)[0]["generated_text"]
    reasoning_log.append({"step": "code_generation", "response": step2_response})

    # Step 3: Mock code review
    review_prompt = (
        f"Review this code for potential logic or syntax issues. Suggest one improvement if possible.\n\nCode:\n{step2_response}"
    )
    review_response = llm(review_prompt, max_new_tokens=150, do_sample=False)[0]["generated_text"]
    reasoning_log.append({"step": "code_review", "response": review_response})

    total_time = round(time.time() - start_time, 2)

    print("🧠 Autonomous developer reasoning completed successfully!\n")
    return reasoning_log, total_time


# === Run Experiment ===
TASK = (
    "Write a Python function to calculate Fibonacci numbers using recursion. "
    "Then print the first 10 numbers and ensure the code is optimized for readability."
)
print(f"💻 Starting OpenDevin-style coding task:\n{TASK}\n")

logs, exec_time = autonomous_dev_agent(TASK)

# === Save Logs ===
print("💾 Saving logs and summary...\n")
with open(LOG_FILE, "w", encoding="utf-8") as f:
    for entry in logs:
        f.write(f"[{entry['step'].upper()}]\n{entry['response']}\n\n")

summary = {
    "model": MODEL_ID,
    "task": TASK,
    "execution_time_sec": exec_time,
    "steps_completed": len(logs),
    "load_time_sec": load_time,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "final_code_excerpt": logs[1]["response"][:300],
    "review_summary": logs[-1]["response"][:250],
}
with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)

print("✅ All files saved successfully!")
print(f"📂 Reasoning Log: {LOG_FILE}")
print(f"📂 Summary JSON: {SUMMARY_FILE}")
print("=======================================================")
print("🎯 Task Completed: OpenDevin-style autonomous developer workflow executed successfully.")
print("=======================================================")
