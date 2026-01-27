# FSEval – Feature Selection Evaluation Suite

**FSEval** is a lightweight, modular Python library designed to **benchmark feature selection and feature ranking methods** across multiple datasets using both **supervised** and **unsupervised** downstream evaluation protocols.

It helps researchers and practitioners answer the question:

> "Which feature selection method actually works best for my type of data and task?"

FSEval automates:

- Repeated training & evaluation at different feature subset sizes
- Stochastic method averaging
- Result persistence & incremental updates
- Support for both classification and clustering-based evaluation

## 📦 Dependencies and Requirements

FSEval requires:

- `python>=3.8`
- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`
- `clustpy` (only needed for `unsupervised_clustering_accuracy`)

## 💡 Installation
You can just download the source code and import fseval, or you can install it using pip:

```bash
pip install sdufseval
```

## 🚀 Quick Example

```python
from sdufseval import FSEVAL
import numpy as np

if __name__ == "__main__":

    # The 23 benchmark datasets
    DATASETS_TO_RUN = [
        'ALLAML', 'CLL_SUB_111', 'COIL20', 'Carcinom', 'GLIOMA', 'GLI_85', 
        'Isolet', 'ORL', 'Prostate_GE', 'SMK_CAN_187', 'TOX_171', 'Yale', 
        'arcene', 'colon', 'gisette', 'leukemia', 'lung', 'lung_discrete', 
        'madelon', 'orlraws10P', 'pixraw10P', 'warpAR10P', 'warpPIE10P'
    ]

    # Initialize FSEVAL
    evaluator = FSEVAL(output_dir="benchmark_results", avg_steps=10)

    # Configuration for methods
    methods_list = [
        {
            'name': 'Random', 
            'stochastic': True, 
            'func': evaluator.random_baseline
        },
        {
            'name': 'Variance_Baseline', 
            'stochastic': False, 
            'func': lambda X: np.var(X, axis=0)
        }
    ]
    
    # --- 1. Run Standard Benchmark ---
    # Evaluates methods on real-world datasets across different feature scales
    evaluator.run(DATASETS_TO_RUN, methods_list)

    # --- 2. Run Runtime Analysis ---
    # Performs scalability testing on synthetic data with a time cap.
    # vary_param='both' triggers both 'features' and 'instances' experiments.
    print("\n>>> Starting Scalability Analysis...")
    evaluator.timer(
        methods=methods_list, 
        vary_param='both', 
        time_limit=3600  # 1 hour limit 
    )
```

## Data Loading

load_dataset(dataset_name, data_dir="datasets") supports:
- Single .mat file with keys 'X' and 'Y'
- Two CSV files: {name}_X.csv and {name}_y.csv

## 📚 API Reference

### 🛠️ `FSEval(output_dir="results", cv=5, avg_steps=10, eval_type="both", metrics=None, experiments=None)`

Initializes the evalutation and benchmark object.

| Parameter | Default | Description |
| :--- | :--- | :--- |
| **`output_dir`** | results | Folder where CSV result files are saved. |
| **`cv`** | 5 | Cross-validation folds (supervised only). |
| **`avg_steps`** | 10 | Number of repetitions for stochastic methods.|
| **`supervised_iter`** | 5 | Number of classifier's runs with different random seeds.|
| **`unsupervised_iter`** | 10 | Number of clustering runs with different random seeds.|
| **`eval_type`** | both | "supervised", "unsupervised", or "both". |
| **`metrics`** | ["CLSACC", "NMI", "ACC", "AUC"] | Evaluation metrics to calculate. |
| **`experiments`** | ["10Percent", "100Percent"] | Which feature ratio grids to evaluate. |

### ⚙️ `run(datasets, methods, classifier=None)`

Initializes the evalutation and benchmark object.

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`datasets`** | List[str] | Dataset names loadable via load_dataset(). |
| **`methods`** | List[dict] | "[{""name"": str, ""func"": callable, ""stochastic"": bool}, ...]" |
| **`classifier`** | sklearn classifier | Classifier for supervised eval (default: RandomForestClassifier) |

### ⚙️ `timer(methods, vary_param='features', time_limit=3600)`

Runs a runtime analysis on the methods.

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`methods`** | List[dict] | "[{""name"": str, ""func"": callable, ""stochastic"": bool}, ...]" |
| **`vary_param`** | ["CLSACC", "NMI", "ACC", "AUC"] | "features", "instances", or "both". |
| **`time_limit`** | 3600 | Terminate the method after reecording first time it exceeds this limit. |

#  Dashboard

There is a Feature Selection Evaluation Dashboard based on the benchmarks provided by FSEVAL, available on:

https://fseval.imada.sdu.dk/

The dashboard offers a collection of useful analytic tools to provide comprehensive and comparative insights into the performance of your feature selection method(s).

#  Citation

If you use FSEVAL in your research, please cite the original paper:

```
CITATION WILL BE PROVIDED UPON PUBLICATION.
```
