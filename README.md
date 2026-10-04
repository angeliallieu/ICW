# ML Pipeline Comparison: Vanilla vs. Kedro vs. Prefect

**Vergleich von drei Orchestrierungs-Ansätze für eine Classification Pipeline.**

---

## 🚀 Quick Start

### Installation
```bash
git clone <repo>
cd ICW
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run All Pipelines
```bash
python scripts/compare_pipelines.py
```

**Output:**
```
[VANILLA]  ✓ Success | Runtime: 3.66s | Accuracy: 0.9649
[KEDRO]    ✓ Success | Runtime: 0.50s | Accuracy: 0.9649
[PREFECT]  ✓ Success | Runtime: 18.88s | Accuracy: 0.9649
```

---

## 📋 Commands

### Vanilla Pipeline (Framework-free)
```bash
python -m src.pipelines.vanilla.runner
```

### Kedro Pipeline
```bash
# Nodes direkt nutzen
python scripts/run_kedro_pipeline.py
```

### Prefect Pipeline 
```bash
python -m src.pipelines.prefect_pipeline.flows
```
---

## 🧪 Tests
```bash
# All tests
pytest 
```

---

## 📊 Example Results Comparison

| Framework | Runtime | Accuracy | LOC | Setup Complexity |
|-----------|---------|----------|-----|------------------|
| **Vanilla** | 3.66s | 0.9649 | 275 | Minimal |
| **Kedro** | 0.50s | 0.9649 | 449 | Moderate |
| **Prefect** | 18.88s | 0.9649 | 508 | High |

**Key Insight:** Alle nutzen **identical Core-Funktionen** aus `src/core/`

---

## 🏗️ Project Structure

```
src/
├── core/                    ← Framework-agnostic ML code 
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   └── evaluation.py
│
└── pipelines/
    ├── vanilla/             
    │   └── runner.py
    ├── kedro_pipeline/      
    │   ├── nodes.py
    │   └── pipeline.py
    └── prefect_pipeline/   
        ├── flows.py
        └── __init__.py

config/
├── parameters.yml           ← Hyperparameters (used by all 3)
└── catalog.yml              ← Kedro Data Catalog

data/
├── 01_raw/                  ← Original dataset
├── 02_intermediate/         ← Cleaned & normalized
├── 03_primary/              ← Features & splits
├── 04_models/               ← Trained model
└── 05_metrics/              ← Results & evaluation

tests/                        
├── test_data_processing.py
├── test_feature_engineering.py
├── test_modeling.py
├── test_evaluation.py
├── test_integration.py      ← Vanilla vs Kedro vs Prefect comparison
└── test_prefect_pipeline.py
```

---

## Troubleshooting

### ModuleNotFoundError
```bash
export PYTHONPATH=$(pwd)
python -m src.pipelines.vanilla.runner
```

### Test Failures
```bash
pip install -r requirements.txt
pytest
```

### Prefect Server Issues
```bash
pip install --upgrade prefect==3.0.13
python -m src.pipelines.prefect_pipeline.flows
```

---

## 📚 Resources

- **Kedro:** https://kedro.readthedocs.io/
- **Prefect:** https://www.prefect.io/
- **Scikit-Learn:** https://scikit-learn.org/
- **MLOps:** https://cloud.google.com/architecture/devops-culture/mlops-principles
