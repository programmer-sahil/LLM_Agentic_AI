# ️ Environment Setup Guide for LLM Thesis  
**Project Title:** *From Transformers to Agents – A Comparative Study of LLM and Agentic Frameworks*  
**Author:** SK Sahil  
**Thesis Context:** Bachelor's Thesis – IU International University of Applied Sciences  
**Environment Folder:** `/env/`

---

##  1️⃣ Create a Virtual Environment

### ▶️ Using `venv` (Recommended)
```bash
python3 -m venv env/venv
source env/venv/bin/activate  # macOS / Linux
env\venv\Scripts\activate     # Windows
```

### ▶️ Using Conda

```bash
conda env create -f env/environment.yml
conda activate llm-thesis
```


Tip: Use venv for lightweight environments on local systems (macOS/Linux),
and Conda for complex dependency management or GPU-enabled configurations.


## 2️⃣ Install Dependencies


### ▶️ For pip users:
```bash
conda env create -f env/environment.yml
conda activate llm-thesis
```

### ▶️ For Conda users:
```bash
conda env create -f env/environment.yml
conda activate llm-thesis
```

Note: Ensure your internet connection is stable during installation,
as large model libraries (e.g., Hugging Face Transformers) may take several minutes to download.



## 3️⃣ Verify the Installation

Once dependencies are installed, verify the environment setup with:

```bash
python --version
pip list
jupyter notebook
```
If all libraries load successfully, your environment is ready to execute the following notebooks:

```bash
01_transformers_inference.ipynb
02_agentic_frameworks_experiments.ipynb
03_model_comparison_analysis.ipynb
04_visualization_results.ipynb
```

Successful imports will confirm that your system is properly configured for model inference,
analysis, and visualization tasks.


## 4️⃣ Deactivate the Environment

When finished working:
```bash
deactivate
```

If using Conda:
```bash
conda deactivate
```
Always deactivate your virtual environment after each session
to avoid dependency conflicts between projects.


### Notes & Recommendations

* Tested Environment: Python 3.12

* Supported OS: macOS (Sonoma 14+), Linux, Windows 10/11

* GPU Compatibility: MPS (Apple Silicon), CUDA (NVIDIA), or CPU fallback

* Recommended IDE: VS Code or PyCharm 
  * * Extensions: Python, Jupyter, and Markdown Preview Enhanced

* Storage Recommendation: Minimum 10 GB free space for model weights and results





## 5️⃣ Expected Folder Structure

Your environment directory should appear as follows:

###  Final Environment Folder Structure

```bash
env/
├── venv/
│   ├── bin/
│   ├── lib/
│   └── pyvenv.cfg
├── environment.yml
├── requirements.txt
└── setup_instructions.md
```

Ensure this structure is maintained for compatibility with project scripts
such as inference_tinyllama.py and compare_latency.py.


### Final Verification

Once setup is complete, run the following quick test inside your activated environment:

```bash
python -c "from transformers import pipeline; print('✅ Transformers installed correctly!')"
```
If you see this message, your environment is fully operational 