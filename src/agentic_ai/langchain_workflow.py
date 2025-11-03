"""
====================================================================
File: langchain_workflow.py
Author: SK Sahil
Project: Bachelor's Thesis – From Transformers to Agents
Experiment: LangChain-Style Reasoning Workflow (Tool + Memory)
====================================================================
Description:
Simulates an agent reasoning pipeline inspired by LangChain.
The agent uses:
  - An internal knowledge memory
  - A simple calculator tool
  - Chain-of-thought reasoning steps

Outputs:
  • Step-by-step reasoning log
  • Final response
  • JSON summary for visualization

Saved files:
  /results/agentic_logs/langchain_reasoning_log.txt
  /results/agentic_logs/langchain_session_summary.json
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

LOG_FILE = RESULTS_DIR / "langchain_reasoning_log.txt"
SUMMARY_FILE = RESULTS_DIR / "langchain_session_summary.json"


# === Configuration ===
MODEL_ID = "microsoft/phi-2"
print(f"\n Loading model: {MODEL_ID} ...")

start_load = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")
llm = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
load_time = round(time.time() - start_load, 2)

print(f" Model loaded successfully in {load_time} seconds.\n")


# === Define Tool (Simple Math Tool) ===
def calculator(expression: str):
    """Executes a safe arithmetic expression."""
    try:
        result = eval(expression, {"__builtins__": {}})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


# === Define Memory ===
memory = []  # Store conversation history (prompt + response)


# === Define the Agent ===
def langchain_agent(task: str):
    """
    Simulates a LangChain-style reasoning agent.
    Uses memory and calculator tool to solve tasks step-by-step.
    """
    reasoning_log = []
    start_time = time.time()

    # Step 1: Understand the problem
    step1_prompt = (
        f"You are a reasoning agent. Understand and plan how to solve this:\nTask: {task}\n"
        f"Explain your reasoning process step-by-step."
    )
    step1_response = llm(step1_prompt, max_new_tokens=180, do_sample=False)[0]["generated_text"]
    reasoning_log.append({"step": "problem_analysis", "response": step1_response})
    memory.append({"task": task, "thought": step1_response})

    # Step 2: Tool usage (if math expression present)
    if any(op in task for op in ["+", "-", "*", "/", "%"]):
        tool_response = calculator(task)
        reasoning_log.append({"step": "tool_use", "response": tool_response})
        memory.append({"tool_result": tool_response})
    else:
        tool_response = "No math tool required for this query."

    # Step 3: Generate final reasoning using memory
    context = "\n".join([m["thought"] for m in memory if "thought" in m])
    step3_prompt = (
        f"Use the following context to provide a clear final answer:\n{context}\n"
        f"Tool Output: {tool_response}\n"
        f"Now return a well-structured final response."
    )
    step3_response = llm(step3_prompt, max_new_tokens=200, do_sample=False)[0]["generated_text"]
    reasoning_log.append({"step": "final_response", "response": step3_response})
    memory.append({"final": step3_response})

    total_time = round(time.time() - start_time, 2)
    print(" LangChain-style reasoning completed successfully!\n")

    return reasoning_log, total_time


# === Run Experiment ===
TASK = "Calculate 24 / 3 + 12, then explain how reasoning agents use tools to enhance LLM decision-making."
print(f" Starting reasoning task:\n{TASK}\n")

logs, exec_time = langchain_agent(TASK)

# === Save Logs ===
print(" Saving logs and summary...\n")
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
    "final_response_excerpt": logs[-1]["response"][:300],
}
with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)

print(" All files saved successfully!")
print(f" Reasoning Log: {LOG_FILE}")
print(f" Summary JSON: {SUMMARY_FILE}")
print("=======================================================")
print(" Task Completed: LangChain-style reasoning workflow executed successfully.")
print("=======================================================")
