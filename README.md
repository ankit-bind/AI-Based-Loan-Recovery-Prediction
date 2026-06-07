# AI-Based Loan Recovery Prediction System

> An end-to-end machine learning system that predicts the probability of successful loan recovery for delinquent or at-risk borrowers — combining gradient-boosted models, SHAP explainability, and an interactive Streamlit dashboard for enterprise decision support.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Project Goals](#project-goals)
- [Dashboard Pages](#dashboard-pages)
- [Project Architecture](#project-architecture)
- [ML Pipeline](#ml-pipeline)
- [Notebooks](#notebooks)
- [Key Features Engineered](#key-features-engineered)
- [Model Evaluation Metrics](#model-evaluation-metrics)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the App](#running-the-app)
- [Configuration](#configuration)
- [Author](#author)

---

## Overview

**Recovery Intelligence** is a production-grade AI decision-support system built for banks and financial institutions. It ingests borrower financial data, bureau credit history, repayment patterns, and social risk indicators to produce a **recovery probability score**, a **risk tier classification** (Critical / Low / Moderate / High), and a **recommended next action** for every borrower — all explainable via SHAP.

The system is deployed as a **multi-page Streamlit web application** with support for single-borrower assessment, batch portfolio processing (up to 10,000 rows), model health monitoring, SHAP-based explainability, and interactive business strategy simulation.

---

## Problem Statement

Financial institutions manage thousands of delinquent loans with limited recovery staff. Traditional collection strategies rely on manual review and generic rule-based scoring, which leads to:

- **Missed recoveries** — high-probability accounts slip through due to poor prioritization
- **Wasted operational effort** — officers spend time on borrowers who will never pay
- **Inconsistent decisions** — no standardized framework across collection teams
- **Regulatory risk** — opaque predictions without audit trails or plain-language explanations
- **Slow response** — manual analysis takes hours while recovery windows close fast

Without knowing *which loans are likely to recover*, collection teams work blind and resources are systematically misallocated.

---

## Project Goals

| # | Goal | Description |
|---|------|-------------|
| 1 | **Predict Recovery Probability** | Assign each borrower an ML-derived score indicating likelihood of loan recovery |
| 2 | **Portfolio Prioritization** | Rank and tier the entire loan portfolio by risk so teams always focus where it matters |
| 3 | **Explainable AI Decisions** | Use SHAP to explain every prediction in plain language for transparency and audit |
| 4 | **Strategy Optimization** | Simulate business-risk tradeoffs by adjusting thresholds and measuring monetary impact |
| 5 | **Reduce Default Losses** | Improve recovery rates by surfacing high-probability accounts that would otherwise be missed |
| 6 | **Real-time Decision Support** | Process single borrowers or entire portfolios in seconds, not hours |

---

## Dashboard Pages

The Streamlit app has **7 pages** accessible from the sidebar:

### 🏠 Home (`app.py`)
Landing page with navigation cards to all modules and a full platform overview — including problem statement, solution walkthrough, capabilities, target audience, and tech stack.

### 📊 Portfolio (`1_Portfolio.py`)
Executive banking overview. Displays real-time portfolio health metrics: total borrowers, recovery rate, default rate, critical account count, and AI-generated insights. Top critical accounts are surfaced automatically with one-click actions.

### 👤 Single Borrower (`2_Single_Borrower.py`)
Enter a borrower's financial profile and receive an instant recovery probability assessment. Outputs a borrower intelligence card showing:
- Recovery probability score
- Risk tier badge (Critical / Low / Moderate / High)
- Top 3 SHAP risk factors with directional impact bars
- Confidence level
- Recommended next action (Escalate / Call / Review / Continue)

### 📁 Portfolio Analysis (`3_Portfolio_Analysis.py`)
Upload a CSV of loan accounts and process up to **10,000 borrowers** in a single batch. Results are risk-sorted with critical accounts flagged first. Downloadable reports are generated with all predictions and risk tiers.

### 🩺 Model Health (`4_Model_Health.py`)
Live model performance monitoring. Tracks ROC-AUC, Precision, Recall, and F1 metrics against baseline benchmarks. Compares LightGBM vs XGBoost and displays confusion matrices with False Negative highlighting.

### 🔍 Decision Intelligence (`5_Decision_Intelligence.py`)
SHAP-powered explainability dashboard. Includes:
- SHAP summary plots (global feature importance)
- Bar charts for top contributing features
- Waterfall breakdowns per prediction
- Regulatory-grade audit log entries

### ⚙️ Strategy Config (`6_Strategy_Config.py`)
Interactive business strategy simulator. Adjust the recovery threshold with a live slider and see real-time impact on:
- Flagged account count
- Precision / Recall / F1
- Estimated monetary recovery (₹ Lakhs)
- Officer hours required
- Cost per recovered account

Strategies: **Aggressive** (threshold ≤ 0.15) · **Balanced** (0.15–0.30) · **Conservative** (> 0.30)

### ℹ️ About (`7_About.py`)
Full project documentation page covering the problem, goals, dataset, 8-step workflow, tech stack, and author details.

---

## Project Architecture

```
Data Sources (CSV)
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│                   Training Pipeline                     │
│                                                         │
│  Data Ingestion → Data Validation → Feature Engineering │
│       → Data Transformation → Model Trainer             │
│              → Model Evaluation → Explainability        │
└─────────────────────────────────────────────────────────┘
       │                        │
       ▼                        ▼
  artifacts/               models/
  (stage outputs)     (best_model.pkl,
                       preprocessor.pkl,
                       threshold.pkl,
                       feature_names.json)
       │                        │
       └──────────┬─────────────┘
                  ▼
    ┌─────────────────────────┐
    │   Prediction Pipeline   │
    │  (unified_preprocessor) │
    └─────────────────────────┘
                  │
                  ▼
    ┌─────────────────────────┐
    │   Streamlit Dashboard   │
    │   (7-page multi-app)    │
    └─────────────────────────┘
```

---

## ML Pipeline

The pipeline is orchestrated through `main.py` → `TrainingPipeline` and runs the following stages in order:

### 1. Data Ingestion (`data_ingestion.py`)
- Loads `application_train.csv`, `bureau.csv`, and supplementary tables
- Splits into train / test sets
- Saves raw splits to `artifacts/data_ingestion/`

### 2. Data Validation (`data_validation.py`)
- Schema validation against `config/schema.yaml`
- Missing value analysis and sanity checks
- Outputs validation report to `artifacts/data_validation/`

### 3. Feature Engineering (`feature_engineering.py`)
- Constructs 30+ derived features across financial, bureau, behavioral, and social risk domains
- Bureau aggregation (active/closed loan ratios, credit utilization, overdue metrics)
- Saves engineered dataset to `artifacts/feature_engineering/`

### 4. Data Transformation (`data_transformation.py`)
- Encoding: Binary, One-Hot, Frequency encoding for categoricals
- Scaling and imputation via `sklearn` pipelines
- Serializes `preprocessor.pkl` to `artifacts/data_transformation/`

### 5. Model Trainer (`model_trainer.py`)
- Trains **LightGBM** and **XGBoost** classifiers
- Hyperparameter tuning via `params.yaml`
- Generates ROC curves, confusion matrices, PR curves, feature importance CSVs
- Saves comparison report and best model to `artifacts/model_trainer/`

### 6. Model Evaluation (`model_evaluation.py`)
- Evaluates on held-out test set
- Produces `eval_report.json` (ROC-AUC, precision, recall, F1, threshold)
- Copies final assets to `models/` for dashboard consumption

### 7. Explainability (`explainability.py`)
- Computes SHAP values using TreeExplainer
- Generates SHAP summary plot and waterfall plot
- Saves to `reports/explainability/`

### Prediction Pipeline (`prediction_pipeline.py`)
- `UnifiedPreprocessor` handles live inference for both single and batch predictions
- Applies same feature engineering and transformation as training
- Returns probability score, risk tier, and top SHAP factors per borrower

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_EDA.ipynb` | Exploratory data analysis on the main application dataset |
| `02_bureau_EDA_cs.ipynb` | Bureau credit history EDA and case study |
| `02_data_cleaning.ipynb` | Missing value treatment, outlier detection, deduplication |
| `03_feature_engineering.ipynb` | Deriving financial, behavioral, and risk features |
| `04_model_training.ipynb` | LightGBM & XGBoost training, evaluation, and comparison |
| `05_explainability.ipynb` | SHAP analysis — summary plots, waterfall, feature attribution |
| `06_model_optimization.ipynb` | Threshold tuning, precision-recall tradeoff, business impact simulation |

---

## Key Features Engineered

### Financial Ratios
| Feature | Description |
|---------|-------------|
| `CREDIT_INCOME_RATIO` | Total credit amount relative to annual income |
| `ANNUITY_INCOME_RATIO` | Loan annuity as fraction of annual income |
| `GOODS_CREDIT_RATIO` | Goods price relative to credit amount |

### Bureau Behavioral Features
| Feature | Description |
|---------|-------------|
| `ACTIVE_LOAN_RATIO` | Share of active bureau loans vs total |
| `CLOSED_LOAN_RATIO` | Share of successfully closed bureau loans |
| `CREDIT_UTILIZATION_RATIO` | Bureau credit usage relative to limit |
| `OVERDUE_PER_LOAN` | Average overdue days per bureau loan |

### Employment & Age Features
| Feature | Description |
|---------|-------------|
| `AGE_YEARS` | Borrower age in years |
| `EMPLOYMENT_YEARS` | Years at current employer |
| `EMPLOYMENT_AGE_RATIO` | Employment stability relative to age |

### Social Risk Features
| Feature | Description |
|---------|-------------|
| `TOTAL_SOCIAL_DEFAULTS` | Combined social circle default count |
| `TOTAL_SOCIAL_OBS` | Total social observations available |

### External Risk Score
| Feature | Description |
|---------|-------------|
| `EXT_SOURCE_MEAN` | Mean of external bureau risk scores (EXT_SOURCE_1/2/3) |

---

## Model Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **ROC-AUC** | Primary model quality metric — area under the receiver operating characteristic curve |
| **Recall** | Fraction of actual recoverable loans correctly identified (minimises missed recoveries) |
| **Precision** | Fraction of flagged accounts that are truly recoverable |
| **F1-Score** | Harmonic mean of precision and recall |
| **Confusion Matrix** | TP / TN / FP / FN breakdown with False Negative emphasis |

> **Note:** Recall is prioritised over precision in this domain — missing a recoverable loan (False Negative) is more costly than flagging a non-recoverable one (False Positive).

---

## Tech Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Language** | Python 3.10+ | Core ML pipeline and dashboard |
| **ML Models** | LightGBM, XGBoost | Gradient-boosted classification |
| **ML Utilities** | Scikit-learn | Preprocessing, pipelines, metrics |
| **Explainability** | SHAP | TreeExplainer for feature attribution |
| **Data** | Pandas, NumPy | Data manipulation and feature engineering |
| **Visualisation** | Plotly, Matplotlib, Seaborn | Interactive and static charts |
| **Dashboard** | Streamlit | Multi-page web application |
| **Config** | PyYAML | `config.yaml` and `params.yaml` driven pipeline |
| **Packaging** | setuptools | Editable install via `setup.py` |
| **Logging** | Python logging | Structured pipeline logs in `logs/` |

---

## Project Structure

```
AI-Based-Loan-Recovery-Prediction/
│
├── app.py                          # Streamlit home page
├── main.py                         # Training pipeline entry point
├── setup.py                        # Package setup
├── requirements.txt                # Python dependencies
│
├── pages/                          # Streamlit multi-page app
│   ├── 1_Portfolio.py
│   ├── 2_Single_Borrower.py
│   ├── 3_Portfolio_Analysis.py
│   ├── 4_Model_Health.py
│   ├── 5_Decision_Intelligence.py
│   ├── 6_Strategy_Config.py
│   └── 7_About.py
│
├── src/loanrecovery/               # Core ML package
│   ├── components/                 # Pipeline stage modules
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── feature_engineering.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │   └── explainability.py
│   ├── pipeline/
│   │   ├── training_pipeline.py    # Full training orchestrator
│   │   ├── prediction_pipeline.py  # Live inference pipeline
│   │   └── unified_preprocessor.py # Shared feature + transform logic
│   ├── config.py                   # Config dataclasses
│   ├── constants.py                # Path constants
│   ├── logger.py                   # Logging setup
│   ├── exception.py                # Custom exception handler
│   └── utils.py                    # Shared utility functions
│
├── notebooks/                      # Jupyter exploration notebooks
│   ├── 01_EDA.ipynb
│   ├── 02_bureau_EDA_cs.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_explainability.ipynb
│   └── 06_model_optimization.ipynb
│
├── config/                         # YAML configuration files
│   ├── config.yaml                 # Paths and artifact dirs
│   ├── params.yaml                 # Model hyperparameters
│   └── schema.yaml                 # Dataset column schema
│
├── utils/                          # Streamlit utility modules
│   ├── styles.py                   # Global CSS injection & page config
│   ├── components.py               # Reusable UI components
│   ├── charts.py                   # Plotly chart builders
│   └── insights.py                 # AI insight generators
│
├── artifacts/                      # Pipeline stage outputs (auto-generated)
│   ├── data_ingestion/
│   ├── data_validation/
│   ├── data_transformation/
│   ├── feature_engineering/
│   ├── model_trainer/
│   └── model_evaluation/
│
├── models/                         # Final production model artifacts
│   ├── best_model.pkl
│   ├── preprocessor.pkl
│   ├── threshold.pkl
│   └── feature_names.json
│
├── data/
│   ├── raw/                        # Source CSV files
│   └── processed/                  # Processed datasets
│
├── reports/
│   └── explainability/             # SHAP plots
│
└── logs/                           # Pipeline execution logs
```

---

## Setup & Installation

### Prerequisites
- Python 3.10 or higher
- pip

### 1. Clone the repository
```bash
git clone https://github.com/ankit-bind/AI-Based-Loan-Recovery-Prediction.git
cd AI-Based-Loan-Recovery-Prediction
```

### 2. Create and activate a virtual environment
```bash
python -m venv myenv
# Windows
myenv\Scripts\activate
# macOS / Linux
source myenv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add raw data
Place the following CSV files in `data/raw/`:
- `application_train.csv`
- `bureau.csv`
- `previous_application.csv`
- `installments_payments.csv`
- `POS_CASH_balance.csv`

> Source: [Home Credit Default Risk — Kaggle](https://www.kaggle.com/c/home-credit-default-risk)

### 5. Run the training pipeline
```bash
python main.py
```
This will execute all pipeline stages and populate `artifacts/` and `models/`.

---

## Running the App

```bash
python -m streamlit run app.py
```

The dashboard will open at `http://localhost:8501`.

> **Pre-trained model required:** The dashboard reads from `models/best_model.pkl` and `artifacts/model_evaluation/eval_report.json`. Run `python main.py` first to generate these files, or use pre-trained artifacts if provided.

---

## Configuration

All pipeline behaviour is controlled via YAML files in `config/`:

| File | Purpose |
|------|---------|
| `config/config.yaml` | Data paths, artifact directories, model output paths, logging and report dirs |
| `config/params.yaml` | Model hyperparameters for LightGBM and XGBoost |
| `config/schema.yaml` | Expected column schema for data validation |

To change model hyperparameters, edit `config/params.yaml` and re-run `python main.py`.

---

## Author

**Ankit**
- Email: itz.ankitbind01@gmail.com
- Project: AI-Based Loan Recovery Probability Prediction System
- Version: 0.0.1

---

> **Disclaimer:** This application provides AI-driven insights for decision support only. All predictions are probabilistic estimates based on historical data patterns. Final loan recovery decisions must include human business review and comply with applicable regulatory requirements.
