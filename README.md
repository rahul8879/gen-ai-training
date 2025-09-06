# Full‑Stack AI/ML/GenAI Training — Rahul Tiwari

Hands‑on repository for a professional, end‑to‑end training covering Python for AI, ML/DL, GenAI, and practical projects. Curated and delivered by Rahul Tiwari.

Profile: https://www.linkedin.com/in/rahul-tiwari-120897/

## Quick Start

- Prerequisites: Python 3.9+ (3.11 recommended), `pip`, `git`, and Jupyter (or VS Code + Python & Jupyter extensions).
- Create environment:
  - `python -m venv .venv && source .venv/bin/activate` (macOS/Linux)
  - `py -3 -m venv .venv && .venv\Scripts\activate` (Windows)
- Install deps: `pip install -r requirements.txt`
- Launch notebooks: `jupyter lab` or `jupyter notebook` (or open in VS Code and select the `.venv` kernel).

## Repository Structure

- `01_PYTHON_FOR_AI/`: Python essentials for AI
  - `00_Python_Essentials_for_AI.ipynb`: professional primer (reproducibility, NumPy, pandas, sklearn pipeline)
  - `01_Basic.ipynb`: additional Python refreshers
- `02_Basic_Math/`: linear algebra, calculus, probability for ML (notebooks and cheatsheets)
- `03_ML/`: classic ML (feature engineering, model selection, evaluation, pipelines)
- `04_DL/`: deep learning (PyTorch/TensorFlow, CNNs, RNNs, Transformers basics)
- `05_GenAI/`: LLMs and GenAI (prompting, fine‑tuning, RAG, evaluation)
- `ai-env/`: optional environment scaffolding or helper scripts
- `.env`: local configuration (API keys, endpoints). Do not commit secrets.
- `requirements.txt`: training dependencies

Note: Folders will be populated progressively as the course advances.

## How To Use This Repo

- Start with `01_PYTHON_FOR_AI/00_Python_Essentials_for_AI.ipynb` to align on environment, patterns, and baseline ML.
- Run notebooks top‑to‑bottom, re‑running cells as needed. Keep a clean kernel per notebook.
- For large datasets, use a `data/` folder (git‑ignored) and provide a short README in that folder with source and schema.
- Prefer vectorized operations (NumPy/pandas) and `sklearn.Pipeline` for leakage‑free baselines.

## Case Studies & Projects (Examples)

- Churn Prediction (tabular): feature engineering, model pipelines, SHAP interpretability
- NLP Classification: text preprocessing, classical ML vs. small transformer finetune
- Vision Transfer Learning: fine‑tune a pretrained CNN with strong baselines
- Time Series Forecasting: baselines vs. tree‑based models; backtesting
- GenAI RAG Mini‑Project: embeddings, vector store, retrieval, evaluation, and guardrails

Each project includes: problem framing, metrics, baseline, iteration, and a short write‑up.

## Professional Practices

- Reproducibility: set seeds, log package versions, pin critical deps when needed.
- Code Quality: type hints, assertions, small functions/utilities; consistent naming.
- Data Handling: keep raw/processed split; document any cleaning steps; version sizable datasets externally.
- Experimentation: track metrics and artifacts; start with strong baselines before tuning.
- Security: store credentials only in `.env` or your secret manager; never hard‑code keys.

## Environment Tips

- VS Code: install Python, Pylance, Jupyter, and isort/black (if desired). Select the `.venv` interpreter.
- Jupyter: use a dedicated kernel per project; restart kernels when changing dependencies.
- macOS M‑series: prefer native arm64 wheels; if mixing, create a fresh `.venv`.

## Getting Help / Contact

- Instructor: Rahul Tiwari — LinkedIn: https://www.linkedin.com/in/rahul-tiwari-120897/
- Open issues for bugs or requests. PRs welcome for typos, clarifications, or small improvements.

## Roadmap (High‑Level)

- Python for AI → Math for ML → Classical ML → Deep Learning → GenAI → MLOps & Deployment → Capstone Projects

If you want me to pin dependencies precisely for your environment or add a setup script, let me know your OS and Python version.
