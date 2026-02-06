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
                 experiments=None,
                 save_all=False):
        self.output_dir = output_dir
        self.cv = cv
        self.avg_steps = avg_steps
        self.supervised_iter = supervised_iter
        self.unsupervised_iter = unsupervised_iter
        self.eval_type = eval_type
        self.save_all = save_all
        
        all_metrics = ["CLSACC", "NMI", "ACC", "AUC"]
        self.selected_metrics = metrics if metrics else all_metrics
        
        self.scales = {}
        target_exps = experiments if experiments else ["10Percent", "100Percent"]
        if "10Percent" in target_exps:
            self.scales["10Percent"] = np.round(np.arange(0.005, 0.1001, 0.005), 3)
        if "100Percent" in target_exps:
            self.scales["100Percent"] = np.round(np.arange(0.05, 1.001, 0.05), 2)
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def random_baseline(self, X, **kwargs):
        return np.random.rand(X.shape[1])

    def _should_skip(self, ds_name, methods):
        for m_info in methods:
            for scale_name in self.scales.keys():
                last_met = self.selected_metrics[-1]
                fname = os.path.join(self.output_dir, f"{m_info['name']}_{last_met}_{scale_name}.csv")
                
                if not os.path.exists(fname):
                    return False
                
                df = pd.read_csv(fname)
                if 'Dataset' not in df.columns or ds_name not in df['Dataset'].values:
                    return False
        return True

    def run(self, datasets, methods, classifier=None):
        warnings.filterwarnings("ignore")
        
        for ds_name in datasets:
            if self._should_skip(ds_name, methods):
                print(f">>> Skipping {ds_name}")
                continue

            X, y_raw = load_dataset(ds_name)
            if X is None: continue
            
            y = pd.Series(y_raw)
            n_features = X.shape[1]

            for m_info in methods:
                name = m_info['name']
                fs_func = m_info['func']
                repeats = self.avg_steps if m_info.get('stochastic', False) else 1
                
                ds_results = {s: {met: [] for met in self.selected_metrics} for s in self.scales}

                for r in range(repeats):
                    print(f"  [{name}] {ds_name} - Run {r+1}/{repeats}")
                    scores = fs_func(X)
                    indices = np.argsort(scores)[::-1]

                    for scale_name, percentages in self.scales.items():
                        row = {met: {'Dataset': ds_name} for met in self.selected_metrics}
                        for p in percentages:
                            k = max(1, min(math.ceil(p * n_features), n_features))
                            X_subset = X[:, indices[:k]]

                            res = {"CLSACC": np.nan, "NMI": np.nan, "ACC": np.nan, "AUC": np.nan}
                            if self.eval_type in ["unsupervised", "both"]:
                                res["CLSACC"], res["NMI"] = unsupervised_eval(X_subset, y, avg_steps=self.unsupervised_iter)
                            if self.eval_type in ["supervised", "both"]:
                                res["ACC"], res["AUC"] = supervised_eval(X_subset, y, classifier=classifier, cv=self.cv, avg_steps=self.supervised_iter)

                            for met in self.selected_metrics:
                                row[met][p] = res[met]
                        
                        for met in self.selected_metrics:
                            ds_results[scale_name][met].append(row[met])

                self._save_results(name, ds_results)

    def _save_results(self, method_name, ds_results):
        for scale, metrics in ds_results.items():
            for met_name, rows in metrics.items():
                df_new = pd.DataFrame(rows)
                
                if not self.save_all:
                    df_new = df_new.groupby('Dataset').mean().reset_index()
                
                df_new.columns = df_new.columns.astype(str)
                fname = os.path.join(self.output_dir, f"{method_name}_{met_name}_{scale}.csv")
                
                if os.path.exists(fname):
                    df_old = pd.read_csv(fname)
                    df_old.columns = df_old.columns.astype(str)
                    
                    if self.save_all:
                        df_final = pd.concat([df_old, df_new], ignore_index=True)
                    else:
                        df_final = pd.concat([df_old, df_new]).drop_duplicates(subset=['Dataset'], keep='last')
                else:
                    df_final = df_new
                
                df_final.to_csv(fname, index=False)

    def timer(self, methods, vary_param='both', time_limit=3600):
        experiments = []
        if vary_param in ['features', 'both']:
            experiments.append({'name': 'features', 'fixed_val': 100, 'range': range(1000, 20001, 500), 'file': 'time_analysis_features.csv'})
        if vary_param in ['instances', 'both']:
            experiments.append({'name': 'instances', 'fixed_val': 100, 'range': range(1000, 20001, 500), 'file': 'time_analysis_instances.csv'})

        for exp in experiments:
            vary_type = exp['name']
            val_range = exp['range']
            filename = os.path.join(self.output_dir, exp['file'])
            timed_out_methods = set()
            results = {m['name']: [] for m in methods}
            
            for val in val_range:
                if vary_type == 'features':
                    n_samples, n_features = exp['fixed_val'], val
                else:
                    n_samples, n_features = val, exp['fixed_val']

                try:
                    X = np.random.rand(n_samples, n_features)
                except MemoryError:
                    for m in methods:
                        results[m['name']].append(-1 if m['name'] in timed_out_methods else np.nan)
                    continue

                for m_info in methods:
                    name = m_info['name']
                    func = m_info['func']
                    if name in timed_out_methods:
                        results[name].append(-1)
                        continue
                    
                    try:
                        start_time = time.time()
                        func(X)
                        duration = time.time() - start_time
                        if duration > time_limit:
                            timed_out_methods.add(name)
                        results[name].append(duration)
                    except Exception:
                        results[name].append(np.nan)

            df_results = pd.DataFrame.from_dict(results, orient='index', columns=list(val_range))
            df_results.index.name = 'Method'
            df_results.to_csv(filename)
