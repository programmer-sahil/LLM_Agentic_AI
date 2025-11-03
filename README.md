#  From Transformers to Agents: A Comparative Analysis of LLM Evolutions and Agentic AI Frameworks

This repository contains all source code, Jupyter notebooks, datasets, and documentation developed for the **Bachelor’s Thesis** by **SK Sahil** at **IU International University of Applied Sciences**.

---

##  Folder Overview

|  Folder |  Description |
|----------------|----------------|
| `docs/` | Contains research references, diagrams, and exported visuals used in the thesis report. |
| `notebooks/` | Jupyter/Colab notebooks for inference, visualization, and agentic AI experiments. |
| `src/` | Core Python source code — includes `llm_evolution` (LLM comparison) and `agentic_ai` (AutoGPT, CrewAI, LangChain, OpenDevin). |
| `results/` | Stores experiment outputs such as model logs, token usage data, latency charts, and visualizations. |
| `appendix/` | Supplementary materials (screenshots, generated figures, Markdown summaries for appendix). |
| `env/` | Environment configuration — includes `requirements.txt`, `environment.yml`, and setup documentation. |


---

## ️ Environment Setup

### Using `venv`
```bash
python3 -m venv env/venv
source env/venv/bin/activate   # macOS/Linux
env\venv\Scripts\activate      # Windows
pip install -r env/requirements.txt
```

### Using Conda
```bash
conda env create -f env/environment.yml
conda activate llm-thesis
```
Use venv for lightweight local runs; prefer Conda for GPU-enabled configurations.

---

##  Tech Stack

| Category | Tools & Libraries |
|----------------|----------------|
| **Language** | Python 3.10+ |
| **LLM Frameworks** | Hugging Face Transformers ≥ 4.43 |
| **Agentic Frameworks** | CrewAI, AutoGPT, LangChain, OpenDevin |
| **Visualization** | Matplotlib, Pandas |
| **Compute Environment** | macOS M1 (MPS) & Google Colab GPU (T4) |

---

##  Key Experiments

| Phase | Script | Purpose |
|----------------|----------------|----------------|
| **Phase 2** | `inference_tinyllama.py` | Run TinyLlama baseline inference on Apple M1 GPU. |
| **Phase 3** | `inference_phi2.py` | Evaluate Phi-2 model for latency and token efficiency. |
| **Phase 4** | `compare_latency.py` | Compare TinyLlama and Phi-2 performance metrics. |
| **Phase 5** | `visualize_latency.py` | Generate visual analytics and scatter plots. |
| **Phase 6** | `tokenizer_stats.py` | Analyze tokenization efficiency and vocabulary size. |
| **Agentic Phase 1–4** | `autogpt_demo.py`, `crewai_pipeline.py`, `langchain_workflow.py`, `open_devin_test.py` | Compare autonomous reasoning and reflection capabilities across frameworks. |

---

## Reproducibility Notes

* Tested on Python 3.12 (macOS) and Python 3.10 (Colab GPU)

* Compatible with MPS, CUDA, and CPU-only environments

* Run scripts directly from the project root:

```bash
python src/llm_evolution/inference_phi2.py
python src/agentic_ai/crewai_pipeline.py
``` 


##  Project Highlights

- Benchmarks **TinyLlama (1.1B)** vs **Phi-2 (2.7B)** using real latency and token metrics.  
- Visualizes **efficiency and reasoning quality** through a structured multi-phase comparison.  
- Implements **four Agentic AI frameworks** to study collaboration, reflection, and autonomy.  
- Provides **structured outputs and appendix-ready figures** for thesis integration.  

---

## 👨‍💻 Author

**SK Sahil**  
*Bachelor of Science in Computer Science*  
**IU International University of Applied Sciences, Germany 🇩🇪**  
**Supervisor:** Dr. Aditya Mushyam  
📍 *Berlin, Germany*  
🔗 [LinkedIn Profile](https://www.linkedin.com/in/programmer-sahil)  
📧 *Contact available upon request*

---

##  License

This repository is intended **for academic and research purposes only.**  
Do **not redistribute** or use commercial models trained using this code without explicit authorization.  

---
