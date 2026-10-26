# FSEVAL: Feature Selection Evaluation Suite

**FSEVAL** is a Python library for **benchmarking feature selection and feature ranking methods** across multiple datasets with **supervised**, **unsupervised**, **model-agnostic** and **custom** evaluation protocols.

It helps researchers and practitioners answer the question:

> "Which feature selection method actually works best for my type of data and task?"

FSEVAL automates:

- Evaluation at a grid of feature subset sizes, built from each method's feature ranking
- Averaging over repetitions for stochastic methods and over random seeds for classifiers and clustering
- Stability analysis of the resulting performance curves
- Parallel execution over (dataset, method) pairs, with crash-safe, resumable result files
- Scalability (runtime) analysis

## 🆕 What's New in 1.3.0

- **Parallel execution** over (dataset, method) pairs and over feature subsets (`n_jobs`), with a progress bar.
- **Resumable runs**: completed (dataset, method) pairs are skipped; a manifest detects when settings changed and reruns only those pairs. Use `force=True` to rerun everything.
- **Crash safety**: a failing pair is reported with its reason and does not abort the run; all writes are atomic.
- **Cleaner console output**: aligned per-item report lines and an undisturbed progress bar; when output is redirected to a log file, the bar is updated every 30 seconds instead of twice per second.
- **Tidy result files**: rows follow the order of the dataset list, columns follow the percentage grid. `refine()` normalises existing result folders, including files from older versions.
- **New options**: `data_dir`, `random_state`, `n_jobs`, `stability`, `save_classifier_results`, `max_subset_gb`, and per-method `params`.
- **Memory-bounded evaluation**: feature subsets are materialised in chunks of at most `max_subset_gb`.
- **scikit-learn 1.9 compatibility**: for multiclass AUC, the SVM is wrapped in `CalibratedClassifierCV` instead of using the deprecated `SVC(probability=True)`. Multiclass SVM scores therefore differ slightly from 1.2.x; do not mix them with results produced by older versions.

## 📦 Dependencies and Requirements

- `python>=3.11`
- `numpy`, `pandas`, `scipy`, `scikit-learn`, `joblib`
- `xgboost` (one of the default classifiers)
- `pcametric` (`AAD` metric)
- `fsdem` (stability analysis)
- `tqdm` (progress bar)

## 💡 Installation

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
    Average proportion of shared nearest neighbors (k=5)
    between the original space and the feature-selected subspace.
    """
    k = min(5, X_orig.shape[0] - 1)

    def get_nn_indices(data, n_neighbors):
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(data)
        _, indices = nbrs.kneighbors(data)
        return indices[:, 1:]

    nn_orig = get_nn_indices(X_orig, k)
    nn_sub = get_nn_indices(X_sub, k)
    intersections = [len(np.intersect1d(nn_orig[i], nn_sub[i])) for i in range(len(nn_orig))]
    return np.mean(intersections) / k

if __name__ == "__main__":  # required, FSEVAL uses multiprocessing

    DATASETS_TO_RUN = ['colon', 'leukemia', 'prostate_GE']

    evaluator = FSEVAL(
        output_dir="benchmark_results",
        data_dir="datasets",
        avg_steps=5,
        random_state=0,
        n_jobs=-1,
        eval_type=["supervised", "unsupervised", "model_agnostic", "custom"],
        custom_metrics={"SNN_K5": snn_consistency_k5},
    )

    methods_list = [
        {
            'name': 'Random',
            'type': 'unsupervised',
            'stochastic': True,
            'func': evaluator.random_baseline,
        },
        {
            'name': 'Variance_Baseline',
            'type': 'unsupervised',
            'stochastic': False,
            'func': lambda X: np.var(X, axis=0),
        },
    ]

    print(">>> Starting Integrated Evaluation...")
    evaluator.run(DATASETS_TO_RUN, methods_list)

    print("\n>>> Starting Scalability Analysis...")
    evaluator.timer(methods=methods_list, vary_param='both', time_limit=3600)
```

Running the script again skips every (dataset, method) pair that is already complete, so an interrupted benchmark can simply be restarted.

## 🧩 Defining Methods

Each method is a dictionary:

| Key | Required | Description |
| :--- | :--- | :--- |
| **`name`** | yes | Method name, used in the result file names. |
| **`type`** | yes | `"unsupervised"` (called as `func(X, **params)`) or `"supervised"` (called as `func(X, y, **params)`). |
| **`func`** | yes | Callable returning one score per feature; higher scores mean more important features. NaN scores are ranked last. |
| **`stochastic`** | no (False) | If True, the method is run `avg_steps` times with seeds `random_state + r` and the results are averaged. |
| **`params`** | no ({}) | Extra keyword arguments passed to `func`. |

For each subset size `p`, the top `k = ceil(p * n_features)` features of the ranking are evaluated (at least 1, at most `n_features`).

## 📊 Evaluation Protocols

| `eval_type` | Metrics | Protocol |
| :--- | :--- | :--- |
| **`supervised`** | `ACC`, `AUC` | Stratified `cv`-fold cross-validation, repeated with `supervised_iter` seeds. By default the scores are averaged over five classifiers: SVM (RBF), KNN (k=5), XGBoost, Random Forest and Decision Tree. Multiclass AUC is one-vs-rest. |
| **`unsupervised`** | `CLSACC`, `NMI` | K-means with as many clusters as classes, repeated with `unsupervised_iter` seeds. `CLSACC` is the clustering accuracy under the best one-to-one cluster-to-class matching (Hungarian method). |
| **`model_agnostic`** | `AAD` | PCA-based angular deviation between the full data and the selected subset, computed without labels or a learning model (`pcametric`). |
| **`custom`** | keys of `custom_metrics` | Each function is called as `f(X_full, X_subset, y)` and must return a number. |

If `stability` is enabled, the stability of each metric curve over the subset sizes is computed with `fsdem` and saved in separate files.

## 🗂️ Data Loading

`load_dataset(dataset_name, data_dir="datasets", verbose=True)` supports:

- A single `{name}.mat` file with keys `X` and `Y`
- Two CSV files, `{name}_X.csv` and `{name}_y.csv`, each with a header row

`X` must be 2D and have as many rows as `y`. During `run()`, datasets are loaded silently; a dataset that cannot be loaded is reported as a failed item with the reason, and the run continues.

## 📁 Output Files

All results are written to `output_dir`. Each file has one row per dataset (in the order of the dataset list) and one column per subset percentage.

| File | Content |
| :--- | :--- |
| `{method}_{metric}_{experiment}_{signature}.csv` | Metric values per subset size. |
| `{method}_{metric}_Stability_{experiment}_{signature}.csv` | Stability of the metric curve per dataset. |
| `{method}_{classifier}_{metric}_{experiment}_{signature}.csv` | Per-classifier `ACC`/`AUC` (only with `save_classifier_results=True`). |
| `time_analysis_features.csv`, `time_analysis_instances.csv` | Runtimes from `timer()`. |

The signature records the settings the values depend on: `SI{supervised_iter}_CV{cv}_RS{random_state}` for `ACC`/`AUC`, `USI{unsupervised_iter}_RS{random_state}` for `CLSACC`/`NMI`, and `RS{random_state}` otherwise. The hidden file `.fseval_manifest.json` stores the settings each (dataset, method) result was produced with. If saving a result fails, it is kept in `_unsaved/` as a pickle.

Do not run two FSEVAL processes into the same `output_dir` at the same time.

## 📚 API Reference

### 🛠️ `FSEVAL(...)`

Initializes the evaluation and benchmark object.

| Parameter | Default | Description |
| :--- | :--- | :--- |
| **`output_dir`** | `"results"` | Folder where result files are saved. |
| **`data_dir`** | `None` | Folder containing the datasets; `None` uses `"datasets"`. |
| **`cv`** | 5 | Cross-validation folds (supervised only). |
| **`avg_steps`** | 10 | Number of repetitions for stochastic methods. |
| **`supervised_iter`** | 5 | Number of cross-validation runs with different random seeds. |
| **`unsupervised_iter`** | 10 | Number of clustering runs with different random seeds. |
| **`random_state`** | 0 | Base seed for methods, classifiers, cross-validation and clustering. |
| **`n_jobs`** | -1 | Number of CPU cores to use; -1 uses all cores. |
| **`eval_type`** | `["unsupervised", "supervised", "model_agnostic"]` | Any of `"supervised"`, `"unsupervised"`, `"model_agnostic"`, `"custom"`. |
| **`metrics`** | `None` | Metrics to save; `None` uses all metrics of the selected `eval_type`s. When given explicitly, include the names of custom metrics. |
| **`stability`** | `True` | `True` for all selected metrics, a list of metric names, or `False` to disable. |
| **`custom_metrics`** | `None` | Dictionary `{name: f(X_full, X_subset, y)}`. |
| **`experiments`** | `["10Percent", "100Percent"]` | Subset grids: `"10Percent"` is 0.5% to 10% in steps of 0.5%, `"100Percent"` is 5% to 100% in steps of 5%. |
| **`save_all`** | `False` | Save every repetition of stochastic methods as a separate row instead of the mean. |
| **`save_classifier_results`** | `False` | Also save `ACC`/`AUC` for each classifier separately. |
| **`max_subset_gb`** | 2.0 | Upper bound on the memory one worker uses for feature subset copies at once. |

### ⚙️ `run(datasets, methods, classifier=None, force=False)`

Runs the benchmark.

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`datasets`** | List[str] | Dataset names loadable via `load_dataset()`. |
| **`methods`** | List[dict] | Method definitions, see [Defining Methods](#-defining-methods). |
| **`classifier`** | sklearn classifier | Single classifier to use instead of the five default classifiers. |
| **`force`** | bool | Rerun all pairs, including completed ones. |

The manifest identifies a method by its name, type, function name and `params`. If you change the code of a method without renaming it, rerun it with `force=True`.

### ⚙️ `refine(datasets=None)`

Normalises every result file in `output_dir`: merges duplicated columns left by older versions, sorts percentage columns, and orders rows like `datasets` (default: the list of the last `run()`). Datasets not in the list are kept and placed last. `run()` calls it automatically.

### ⚙️ `timer(methods, vary_param="both", time_limit=3600)`

Runs a runtime analysis on random data: 100 instances with 1,000 to 20,000 features, and/or 100 features with 1,000 to 20,000 instances (step 500).

| Argument | Default | Description |
| :--- | :--- | :--- |
| **`methods`** | | Method definitions, see [Defining Methods](#-defining-methods). Supervised methods receive a 5-class label vector. |
| **`vary_param`** | `"both"` | `"features"`, `"instances"`, or `"both"`. |
| **`time_limit`** | 3600 | Once a method exceeds this limit (in seconds), it is skipped for larger sizes and recorded as -1. A running call is not interrupted. |

### 🔧 Standalone Functions

| Function | Description |
| :--- | :--- |
| **`supervised_eval_batch(X_subsets, y, classifier=None, cv=5, avg_steps=10, n_jobs=-1, random_state=42)`** | Evaluates a dictionary `{name: X}` of feature subsets; returns `{name: {classifier: {"acc", "auc"}}}`. |
| **`unsupervised_eval_batch(X_subsets, y, avg_steps=10, n_jobs=-1, random_state=42)`** | Evaluates a dictionary `{name: X}` of feature subsets with K-means; returns `{name: {"clsacc", "nmi"}}`. |
| **`clustering_accuracy(labels_true, labels_pred)`** | Clustering accuracy under the best one-to-one matching between clusters and classes. |
| **`load_dataset(dataset_name, data_dir="datasets", verbose=True)`** | Returns `(X, y)`, or `(None, None)` if the dataset cannot be loaded. `verbose` prints loading messages. |

# Dashboard

There is a Feature Selection Evaluation Dashboard based on the benchmarks provided by FSEVAL, available on:

https://fseval.imada.sdu.dk/

The dashboard offers a collection of analytic tools that provide comprehensive and comparative insights into the performance of your feature selection method(s). You can also download the dashboard source code from the dashboard folder of the GitHub repository and use it locally.

# Citation

If you use FSEVAL in your research, please cite the original paper:

```
@article{DBLP:journals/corr/abs-2604-18227,
  author       = {Muhammad Rajabinasab and
                  Arthur Zimek},
  title        = {{FSEVAL:} Feature Selection Evaluation Toolbox and Dashboard},
  journal      = {CoRR},
  volume       = {abs/2604.18227},
  year         = {2026},
  url          = {https://doi.org/10.48550/arXiv.2604.18227},
  doi          = {10.48550/ARXIV.2604.18227},
  eprinttype   = {arXiv},
  eprint       = {2604.18227}
}
```
