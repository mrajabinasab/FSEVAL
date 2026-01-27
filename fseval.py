import os
import math
import time
import warnings
import numpy as np
import pandas as pd
from eval import unsupervised_eval, supervised_eval
from loader import load_dataset


class FSEVAL:
    def __init__(self, 
                 output_dir="results", 
                 cv=5, 
                 avg_steps=10,
                 supervised_iter=5,
                 unsupervised_iter=10, 
                 eval_type="both", 
                 metrics=None, 
                 experiments=None):
        """
        Feature Selection Evaluation Suite.
        """
        self.output_dir = output_dir
        self.cv = cv
        self.avg_steps = avg_steps
        self.supervised_iter = supervised_iter
        self.unsupervised_iter = unsupervised_iter
        self.eval_type = eval_type
        
        # Metric configuration
        all_metrics = ["CLSACC", "NMI", "ACC", "AUC"]
        self.selected_metrics = metrics if metrics else all_metrics
        
        # Experiment/Scale configuration
        self.scales = {}
        target_exps = experiments if experiments else ["10Percent", "100Percent"]
        if "10Percent" in target_exps:
            self.scales["10Percent"] = np.round(np.arange(0.005, 0.1001, 0.005), 3)
        if "100Percent" in target_exps:
            self.scales["100Percent"] = np.round(np.arange(0.05, 1.001, 0.05), 2)
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def random_baseline(self, X, **kwargs):
        """
        Randomly assigns importance scores to features.
        Internal method for lower-bound baseline.
        """
        return np.random.rand(X.shape[1])

    def run(self, datasets, methods, classifier=None):
        """
        Executes the benchmark for given datasets and FS methods.
        
        Args:
            datasets: List of dataset names.
            methods: List of dicts {'name': str, 'func': callable, 'stochastic': bool}.
            classifier: Optional sklearn classifier instance to pass to supervised_eval.
        """
        warnings.filterwarnings("ignore")
        for ds_name in datasets:
            print(f"\n>>> Benchmarking Dataset: {ds_name}")
            X, y_raw = load_dataset(ds_name)
            if X is None: continue
            
            y = pd.Series(y_raw)
            n_features = X.shape[1]

            for m_info in methods:
                name = m_info['name']
                fs_func = m_info['func']
                # Stochastic methods run 10 times and average
                repeats = self.avg_steps if m_info.get('stochastic', False) else 1
                
                # Internal storage for current dataset results
                ds_results = {s: {met: [] for met in self.selected_metrics} for s in self.scales}

                for r in range(repeats):
                    print(f"  [{name}] Progress: {r+1}/{repeats}")
                    
                    # Get feature ranking
                    scores = fs_func(X)
                    indices = np.argsort(scores)[::-1]

                    for scale_name, percentages in self.scales.items():
                        row = {met: {'Dataset': ds_name} for met in self.selected_metrics}
                        
                        for p in percentages:
                            k = max(1, min(math.ceil(p * n_features), n_features))
                            X_subset = X[:, indices[:k]]

                            # Run evaluators
                            c_acc, nmi, acc, auc = np.nan, np.nan, np.nan, np.nan
                            
                            if self.eval_type in ["unsupervised", "both"]:
                                c_acc, nmi = unsupervised_eval(X_subset, y, avg_steps=self.unsupervised_iter)
                            
                            if self.eval_type in ["supervised", "both"]:
                                # Passes classifier (None or instance) to eval.py
                                acc, auc = supervised_eval(X_subset, y, classifier=classifier, cv=self.cv, avg_steps=self.supervised_iter)

                            # Map metrics to columns
                            mapping = {"CLSACC": c_acc, "NMI": nmi, "ACC": acc, "AUC": auc}
                            for met in self.selected_metrics:
                                row[met][p] = mapping[met]
                        
                        for met in self.selected_metrics:
                            ds_results[scale_name][met].append(row[met])

                # Save/Update results for this method/dataset
                self._save_results(name, ds_results)

    
    def timer(self, methods, vary_param='both', time_limit=3600):
        """
        Runs a standalone runtime analysis experiment with a time cap.
        
        Args:
            methods: List of dicts {'name': str, 'func': callable}.
            vary_param: 'features', 'instances', or 'both'.
            time_limit: Max seconds per method before it is skipped.
        """
        
        # Determine which experiments to run
        experiments = []
        if vary_param in ['features', 'both']:
            experiments.append({
                'name': 'features',
                'fixed_val': 100,
                'range': range(1000, 20001, 500),
                'file': 'time_analysis_features.csv'
            })
        if vary_param in ['instances', 'both']:
            experiments.append({
                'name': 'instances',
                'fixed_val': 100,
                'range': range(1000, 20001, 500),
                'file': 'time_analysis_instances.csv'
            })

        for exp in experiments:
            vary_type = exp['name']
            val_range = exp['range']
            filename = os.path.join(self.output_dir, exp['file'])
            
            # Tracking for this specific experiment
            timed_out_methods = set()
            results = {m['name']: [] for m in methods}
            
            print(f"\n--- Starting Experiment: Varying {vary_type} ---")
            print(f"Time limit: {time_limit}s | Output: {filename}")

            for val in val_range:
                # 1. Generate synthetic data based on vary_param
                if vary_type == 'features':
                    n_samples, n_features = exp['fixed_val'], val
                else:
                    n_samples, n_features = val, exp['fixed_val']

                try:
                    X = np.random.rand(n_samples, n_features)
                except MemoryError:
                    print(f"  FATAL: MemoryError: Failed to allocate {n_samples}x{n_features} data.")
                    for m in methods:
                        results[m['name']].append(-1 if m['name'] in timed_out_methods else np.nan)
                    continue

                # 2. Run each method
                for m_info in methods:
                    name = m_info['name']
                    func = m_info['func']
                    
                    # Check if method has already timed out in this experiment
                    if name in timed_out_methods:
                        results[name].append(-1)
                        continue
                    
                    try:
                        start_time = time.time()
                        
                        # Execute the method (assuming benchmark format)
                        func(X)
                        
                        duration = time.time() - start_time
                        
                        if duration > time_limit:
                            print(f"  - {name:<18}: {duration:.4f}s (TIMEOUT - skipping future runs)")
                            timed_out_methods.add(name)
                        else:
                            print(f"  - {name:<18}: {duration:.4f}s")
                        
                        results[name].append(duration)
                        
                    except Exception as e:
                        print(f"  - {name:<18}: FAILED ({type(e).__name__})")
                        results[name].append(np.nan)

            # 3. Save results to CSV
            try:
                df_results = pd.DataFrame.from_dict(results, orient='index', columns=list(val_range))
                df_results.index.name = 'Method'
                df_results.to_csv(filename)
                print(f"\n--- Results saved to {filename} ---")
            except Exception as e:
                print(f"\n--- FAILED to save results: {e} ---")


    def _save_results(self, method_name, ds_results):
        """Aggregates repeats and saves to disk after each dataset."""
        for scale, metrics in ds_results.items():
            for met_name, rows in metrics.items():
                df_new = pd.DataFrame(rows).groupby('Dataset').mean().reset_index()
                fname = os.path.join(self.output_dir, f"{method_name}_{met_name}_{scale}.csv")
                
                if os.path.exists(fname):
                    df_old = pd.read_csv(fname)
                    df_final = pd.concat([df_old, df_new]).drop_duplicates(subset=['Dataset'], keep='last')
                else:
                    df_final = df_new
                
                df_final.to_csv(fname, index=False)
