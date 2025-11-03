"""
====================================================================
File: crewai_pipeline.py
Author: SK Sahil
Project: Bachelor's Thesis – From Transformers to Agents
Experiment: Multi-Agent Collaboration (CrewAI-Style Pipeline)
====================================================================
Description:
This experiment simulates a collaborative pipeline where two agents
(Planner + Worker) communicate to complete a reasoning-based task.

- The Planner agent breaks the main goal into subtasks.
- The Worker agent executes each subtask using LLM inference.
- The Coordinator manages the workflow and aggregates results.

Outputs:
  • Collaborative dialogue between agents
  • Saved JSON and text logs for thesis appendix

Saved files:
  /results/agentic_logs/crewai_conversation_log.txt
  /results/agentic_logs/crewai_pipeline_summary.json
====================================================================
"""

# === Imports ===
import os
import json
import time
from pathlib import Path
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


# === Setup paths ===
BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "results" / "agentic_logs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = RESULTS_DIR / "crewai_conversation_log.txt"
SUMMARY_JSON = RESULTS_DIR / "crewai_pipeline_summary.json"

# === Configuration ===
MODEL_ID = "microsoft/phi-2"
DEVICE = "cuda"
print(f"\n🔹 Loading model: {MODEL_ID} ...")

start_load = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")
llm = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
load_time = time.time() - start_load
print(f"✅ Model loaded successfully in {load_time:.2f}s\n")


# === Define Agent Functions ===

def planner_agent(goal: str):
    """Generates a list of subtasks from the main goal."""
    prompt = (
        f"You are a planning agent. Break down this goal into 3 clear subtasks:\n\nGoal: {goal}\n\n"
        f"Return a numbered list of subtasks."
    )
    response = llm(prompt, max_new_tokens=180, do_sample=False)[0]["generated_text"]
    return response


def worker_agent(subtask: str):
    """Executes an individual subtask."""
    prompt = (
        f"You are a research assistant. Perform this subtask clearly and concisely:\n{subtask}\n"
        f"Return your completed result."
    )
    response = llm(prompt, max_new_tokens=220, do_sample=False)[0]["generated_text"]
    return response


def coordinator(goal: str):
    """Coordinates planner and worker communication."""
    print("🧩 Starting CrewAI-style multi-agent collaboration...\n")
    conversation_log = []
    start_time = time.time()

    # Phase 1: Planning
    plan = planner_agent(goal)
    print("🧠 Planner Agent generated subtasks.\n")
    conversation_log.append({"role": "planner", "content": plan})

    # Extract subtasks (simple heuristic split)
    subtasks = [line for line in plan.split("\n") if any(str(i) in line for i in range(1, 6))]

    # Phase 2: Worker Execution
    all_results = []
    for idx, subtask in enumerate(subtasks, start=1):
        print(f"⚙️ Worker executing subtask {idx} ...")
        result = worker_agent(subtask)
        conversation_log.append({"role": f"worker_{idx}", "content": result})
        all_results.append(f"Subtask {idx}: {result}\n")
        print(f"✅ Subtask {idx} completed.\n")

    # Phase 3: Synthesis
    summary_prompt = (
        "Combine the following completed subtasks into a single structured report:\n"
        + "\n".join(all_results)
        + "\nReturn a concise final summary."
    )
    final_summary = llm(summary_prompt, max_new_tokens=250, do_sample=False)[0]["generated_text"]
    conversation_log.append({"role": "coordinator", "content": final_summary})

    total_time = round(time.time() - start_time, 2)
    print("🎯 Multi-Agent Collaboration Completed!\n")

    return final_summary, conversation_log, total_time


# === Main Experiment ===
goal = (
    "Explain how multi-agent collaboration frameworks enhance the reasoning ability "
    "and efficiency of transformer-based models in real-world problem solving."
)

final_summary, logs, duration = coordinator(goal)

# === Save Logs ===
print("💾 Saving conversation logs...")
with open(LOG_FILE, "w", encoding="utf-8") as f:
    for entry in logs:
        f.write(f"[{entry['role'].upper()}]\n{entry['content']}\n\n")

summary_data = {
    "model": MODEL_ID,
    "goal": goal,
    "execution_time_sec": duration,
    "log_entries": len(logs),
    "final_summary_excerpt": final_summary[:300],
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
}
with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
    json.dump(summary_data, f, indent=4)

print("✅ Logs and summary saved successfully!")
print(f"📂 Log File: {LOG_FILE}")
print(f"📂 Summary JSON: {SUMMARY_JSON}")
print("=======================================================")
print("🎯 Task Completed: CrewAI-Style Multi-Agent Pipeline executed successfully.")
print("=======================================================\n")
