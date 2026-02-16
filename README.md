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
from sklearn.neighbors import NearestNeighbors

def snn_consistency_k5(X_orig, X_sub, y):
    """
    Calculates the average proportion of shared nearest neighbors (k=5) 
    between the original space and the feature-selected subspace.
    """
    k = 5
    k = min(k, X_orig.shape[0] - 1)
    
    def get_nn_indices(data, n_neighbors):
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='auto').fit(data)
        _, indices = nbrs.kneighbors(data)
        return indices[:, 1:]

    nn_orig = get_nn_indices(X_orig, k)
    nn_sub = get_nn_indices(X_sub, k)
    
    intersections = [len(np.intersect1d(nn_orig[i], nn_sub[i])) for i in range(len(nn_orig))]
    return np.mean(intersections) / k

if __name__ == "__main__":

    DATASETS_TO_RUN = ['colon', 'leukemia', 'prostate_GE']

    evaluator = FSEVAL(
        output_dir="benchmark_results", 
        avg_steps=5,
        eval_type=["supervised, "unsupervised", ""model_agnostic", "custom"],
        custom_metrics={"SNN_K5": snn_consistency_k5}
    )

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
    
    print(">>> Starting Integrated Evaluation (Global & Local metrics)...")
    evaluator.run(DATASETS_TO_RUN, methods_list)

    print("\n>>> Starting Scalability Analysis...")
    evaluator.timer(
        methods=methods_list, 
        vary_param='both', 
        time_limit=3600 
    )
```

## Data Loading

load_dataset(dataset_name, data_dir="datasets") supports:
- Single .mat file with keys 'X' and 'Y'
- Two CSV files: {name}_X.csv and {name}_y.csv

## 📚 API Reference

### 🛠️ `FSEval(output_dir="results", cv=5, avg_steps=10, eval_type=["supervised", "unsupervised", "model_agnostic"], metrics=None, experiments=None)`

Initializes the evalutation and benchmark object.

| Parameter | Default | Description |
| :--- | :--- | :--- |
| **`output_dir`** | results | Folder where CSV result files are saved. |
| **`cv`** | 5 | Cross-validation folds (supervised only). |
| **`avg_steps`** | 10 | Number of repetitions for stochastic methods.|
| **`supervised_iter`** | 5 | Number of classifier's runs with different random seeds.|
| **`unsupervised_iter`** | 10 | Number of clustering runs with different random seeds.|
| **`eval_type`** | ["supervised", "unsupervised", "model_agnostic"] | "supervised", "unsupervised", "model_agnostic", or "custom" to enable inclusion of custom user-defined metrics. |
| **`metrics`** | ["CLSACC", "NMI", "ACC", "AUC", "AAD"] | Evaluation metrics to calculate. |
| **`custom_metrics`** | {} | User-defined custom evaluation metrics. |
| **`experiments`** | ["10Percent", "100Percent"] | Which feature ratio grids to evaluate. |
| **`save_all`** | False | Save the results of all runs of the stochastic methods separately. |

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
