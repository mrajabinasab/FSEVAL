"""
fseval.py: feature-selection evaluation with parallel, crash-safe, tidy result files.

Design notes
------------
* Workers compute only. They return their result tables to the main process,
  which is the single writer of every CSV (no races, no file locks).
* All percentage column labels are strings (see _col_label), so merging with
  existing files never creates duplicate column blocks.
* Every write is atomic (temp file + os.replace).
* Rows in every result file follow the order of the `datasets` list given to
  run() (or refine()). refine() also normalises all existing files post hoc.
* A small manifest (.fseval_manifest.json) records the settings each
  (method, dataset) result was produced with, so changed parameters are
  detected instead of silently skipped.
* Do not run two FSEVAL processes into the same output_dir at the same time.
"""

import os
import sys
import glob
import json
import math
import pickle
import random
import time
import hashlib
import threading
import warnings
from collections import defaultdict
from multiprocessing import Manager

import numpy as np
import pandas as pd
import fsdem
from joblib import Parallel, delayed
from pcametric import AAD
from .eval import unsupervised_eval_batch, supervised_eval_batch
from .loader import _load

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


class FSEVAL:
    MANIFEST = ".fseval_manifest.json"

    def __init__(
        self,
        output_dir="results",
        cv=5,
        avg_steps=10,
        supervised_iter=5,
        unsupervised_iter=10,
        random_state=0,
        n_jobs=-1,
        eval_type=["unsupervised", "supervised", "model_agnostic"],
        metrics=None,
        stability=True,
        custom_metrics=None,
        experiments=None,
        save_all=False,
        save_classifier_results=False,
        data_dir=None,
        max_subset_gb=2.0,
    ):
        self.output_dir = output_dir
        self.cv = cv
        self.avg_steps = avg_steps
        self.supervised_iter = supervised_iter
        self.unsupervised_iter = unsupervised_iter
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.eval_type = eval_type if isinstance(eval_type, list) else [eval_type]
        self.save_all = save_all
        self.save_classifier_results = save_classifier_results
        self.custom_metrics = custom_metrics if custom_metrics else {}
        # Folder the datasets are loaded from.
        self.data_dir = data_dir if data_dir else "datasets"
        # Upper bound for the feature-subset copies one worker holds at once.
        self.max_subset_bytes = int(max_subset_gb * 1024 ** 3)

        self.metric_map = {
            "unsupervised": ["CLSACC", "NMI"],
            "supervised": ["ACC", "AUC"],
            "model_agnostic": ["AAD"],
            "custom": list(self.custom_metrics.keys()),
        }

        if metrics:
            self.selected_metrics = metrics
        else:
            self.selected_metrics = []
            for etype in self.eval_type:
                if etype in self.metric_map:
                    self.selected_metrics.extend(self.metric_map[etype])

        self.stability_metrics = stability
        if self.stability_metrics is True:
            self.stability_metrics = list(self.selected_metrics)
        elif self.stability_metrics is False or self.stability_metrics is None:
            self.stability_metrics = []

        self.scales = {}
        target_exps = experiments if experiments else ["10Percent", "100Percent"]
        if "10Percent" in target_exps:
            self.scales["10Percent"] = np.round(np.arange(0.005, 0.1001, 0.005), 3)
        if "100Percent" in target_exps:
            self.scales["100Percent"] = np.round(np.arange(0.05, 1.001, 0.05), 2)

        os.makedirs(self.output_dir, exist_ok=True)

        # Logical CPUs
        detected = os.cpu_count() or 1
        if self.n_jobs == -1:
            self.n_cores = detected
        else:
            self.n_cores = max(1, min(int(self.n_jobs), detected))

        # Number of subset sizes (for progress accounting)
        self._n_subsets = sum(len(p) for p in self.scales.values())

        # Dataset -> position, taken from the list passed to run() / refine().
        self._ds_order = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def random_baseline(self, X, **kwargs):
        return np.random.rand(X.shape[1])

    def _result_signature(self, metric):
        if metric in ["ACC", "AUC"]:
            return f"SI{self.supervised_iter}_CV{self.cv}_RS{self.random_state}"
        if metric in ["CLSACC", "NMI"]:
            return f"USI{self.unsupervised_iter}_RS{self.random_state}"
        return f"RS{self.random_state}"

    @staticmethod
    def _key(ds_name, method_name):
        return f"{ds_name}|{method_name}"

    @staticmethod
    def _col_label(p):
        """Canonical string header for a percentage (matches what to_csv writes for floats)."""
        return str(float(p))

    def _pct_cols(self, scale_name):
        return [self._col_label(p) for p in self.scales[scale_name]]

    def _metric_path(self, method_name, metric, scale_name):
        return os.path.join(
            self.output_dir,
            f"{method_name}_{metric}_{scale_name}_{self._result_signature(metric)}.csv",
        )

    def _classifier_path(self, method_name, clf_name, metric, scale_name):
        return os.path.join(
            self.output_dir,
            f"{method_name}_{clf_name}_{metric}_{scale_name}_{self._result_signature(metric)}.csv",
        )

    def _stability_path(self, method_name, metric, scale_name):
        return os.path.join(
            self.output_dir,
            f"{method_name}_{metric}_Stability_{scale_name}_{self._result_signature(metric)}.csv",
        )

    @staticmethod
    def _fmt_time(seconds):
        seconds = int(max(0, seconds))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _units_per_repeat(self):
        """How many progress units one stochastic repeat is worth."""
        u = 1  # ranking itself
        if "unsupervised" in self.eval_type:
            u += self._n_subsets
        if "supervised" in self.eval_type:
            # 5 default classifiers x subsets (approx; exact count not critical)
            u += self._n_subsets * 5
        if "model_agnostic" in self.eval_type:
            u += self._n_subsets
        return u

    def _allocate_cores(self, n_work):
        """
        Split logical CPUs between outer workers and inner joblib jobs.
          outer ~ cores/4,  inner = cores/outer,  outer*inner <= cores
        """
        if n_work <= 0:
            return 1, self.n_cores
        if n_work == 1:
            return 1, self.n_cores
        max_outer = max(1, self.n_cores // 4)
        n_outer = min(n_work, max_outer, self.n_cores)
        inner_jobs = max(1, self.n_cores // n_outer)
        return n_outer, inner_jobs

    # ------------------------------------------------------------------
    # Manifest and resume logic
    # ------------------------------------------------------------------

    def _manifest_path(self):
        return os.path.join(self.output_dir, self.MANIFEST)

    def _load_manifest(self):
        try:
            with open(self._manifest_path()) as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_manifest(self, manifest):
        tmp = self._manifest_path() + ".tmp"
        with open(tmp, "w") as f:
            json.dump(manifest, f, indent=1, sort_keys=True)
        os.replace(tmp, self._manifest_path())

    def _fingerprint(self, m_info):
        """Hash of everything that influences a method's results."""
        stochastic = bool(m_info.get("stochastic", False))
        func = m_info.get("func")
        payload = {
            "name": m_info["name"],
            "type": m_info["type"],
            "func": getattr(func, "__qualname__", type(func).__name__),
            "params": m_info.get("params", {}),
            "stochastic": stochastic,
            "avg_steps": self.avg_steps if stochastic else 1,
            "supervised_iter": self.supervised_iter,
            "unsupervised_iter": self.unsupervised_iter,
            "cv": self.cv,
            "random_state": self.random_state,
            "eval_type": sorted(self.eval_type),
            "metrics": list(self.selected_metrics),
            "stability": list(self.stability_metrics),
            "save_all": self.save_all,
            "save_classifier_results": self.save_classifier_results,
            "scales": {k: [self._col_label(p) for p in v] for k, v in self.scales.items()},
        }
        blob = json.dumps(
            payload,
            sort_keys=True,
            default=lambda o: getattr(o, "__qualname__", type(o).__name__),
        )
        return hashlib.sha1(blob.encode()).hexdigest()[:16]

    def _datasets_in(self, fname, cache):
        if fname not in cache:
            try:
                cache[fname] = set(self._read_csv_clean(fname)["Dataset"].astype(str))
            except Exception:
                cache[fname] = set()
        return cache[fname]

    def _method_is_complete(self, ds_name, m_info, manifest, cache):
        """Return (complete, reason)."""
        if not self.selected_metrics:
            return False, "no metrics selected"
        name = m_info["name"]
        for scale_name in self.scales:
            for met in self.selected_metrics:
                fname = self._metric_path(name, met, scale_name)
                if ds_name not in self._datasets_in(fname, cache):
                    return False, "results missing"
        recorded = manifest.get(name, {}).get(ds_name)
        if recorded is None:
            return True, "complete, predates manifest"
        if recorded != self._fingerprint(m_info):
            return False, "settings changed"
        return True, "complete"

    # ------------------------------------------------------------------
    # Single (dataset, method) worker (computes only, never writes CSVs)
    # ------------------------------------------------------------------

    def _run_one(self, ds_name, m_info, *args):
        """Crash-proof wrapper: a failing pair is reported, it does not abort the run."""
        t0 = time.time()
        try:
            return self._run_one_inner(ds_name, m_info, *args)
        except Exception as e:
            return {
                "ds": ds_name,
                "method": m_info["name"],
                "status": "error",
                "detail": f"{type(e).__name__}: {e}",
                "elapsed": time.time() - t0,
            }

    def _subset_chunks(self, subset_info, n_samples, itemsize):
        """Group subset names so one group's copies stay under max_subset_bytes."""
        chunk, size = [], 0
        for name, (_, _, k) in subset_info.items():
            nbytes = n_samples * k * itemsize
            if chunk and size + nbytes > self.max_subset_bytes:
                yield chunk
                chunk, size = [], 0
            chunk.append(name)
            size += nbytes
        if chunk:
            yield chunk

    def _run_one_inner(self, ds_name, m_info, classifier, inner_jobs,
                       progress, key, units_per_repeat):
        warnings.filterwarnings("ignore")
        t0 = time.time()

        # Workers load silently; printing here would break the progress bar.
        X, y_raw, load_msgs = _load(ds_name, self.data_dir)
        if X is None:
            return {
                "ds": ds_name,
                "method": m_info["name"],
                "status": "load_failed",
                "detail": load_msgs[-1],
                "elapsed": time.time() - t0,
            }

        y = pd.Series(np.asarray(y_raw).ravel())
        n_samples, n_features = X.shape
        name = m_info["name"]
        fs_type = m_info["type"]
        fs_func = m_info["func"]
        fs_params = m_info.get("params", {})
        repeats = self.avg_steps if m_info.get("stochastic", False) else 1
        item_total = repeats * units_per_repeat
        issues = set()

        ds_results = {
            s: {met: [] for met in self.selected_metrics} for s in self.scales
        }
        classifier_results_all = {s: {} for s in self.scales}

        for r in range(repeats):
            # Seed the global RNGs so stochastic FS methods are reproducible.
            seed = (self.random_state + r) % (2 ** 32)
            np.random.seed(seed)
            random.seed(seed)

            if fs_type == "unsupervised":
                scores = fs_func(X, **fs_params)
            elif fs_type == "supervised":
                scores = fs_func(X, y, **fs_params)
            else:
                raise ValueError("Feature selection algorithm type is not valid.")

            scores = np.asarray(scores, dtype=float).ravel()
            if scores.shape[0] != n_features:
                raise ValueError(
                    f"{name} returned {scores.shape[0]} scores for {n_features} features"
                )
            nan_mask = np.isnan(scores)
            if nan_mask.any():
                # argsort()[::-1] would rank NaN first; rank them last instead.
                issues.add(f"{int(nan_mask.sum())} NaN scores were ranked last")
                scores = np.where(nan_mask, -np.inf, scores)
            indices = np.argsort(scores)[::-1]

            subset_info = {}
            for scale_name, percentages in self.scales.items():
                for p in percentages:
                    k = max(1, min(math.ceil(p * n_features), n_features))
                    subset_info[f"{scale_name}_{p}"] = (scale_name, p, k)

            unsupervised_results = {}
            supervised_results = {}

            # Evaluate in memory-bounded chunks instead of materialising every subset.
            for names in self._subset_chunks(subset_info, n_samples, X.dtype.itemsize):
                X_chunk = {n: X[:, indices[:subset_info[n][2]]] for n in names}
                if "unsupervised" in self.eval_type:
                    unsupervised_results.update(
                        unsupervised_eval_batch(
                            X_chunk, y,
                            avg_steps=self.unsupervised_iter,
                            n_jobs=inner_jobs,
                            random_state=self.random_state,
                        )
                    )
                if "supervised" in self.eval_type:
                    supervised_results.update(
                        supervised_eval_batch(
                            X_chunk, y,
                            classifier=classifier,
                            cv=self.cv,
                            avg_steps=self.supervised_iter,
                            n_jobs=inner_jobs,
                            random_state=self.random_state,
                        )
                    )
                del X_chunk

            for scale_name, percentages in self.scales.items():
                rows = {met: {"Dataset": ds_name} for met in self.selected_metrics}
                clf_rows = {}

                for p in percentages:
                    subset_name = f"{scale_name}_{p}"
                    k = subset_info[subset_name][2]
                    current_res = {met: np.nan for met in self.selected_metrics}

                    if "unsupervised" in self.eval_type:
                        result = unsupervised_results[subset_name]
                        if "CLSACC" in current_res:
                            current_res["CLSACC"] = result["clsacc"]
                        if "NMI" in current_res:
                            current_res["NMI"] = result["nmi"]

                    if "supervised" in self.eval_type:
                        classifier_res = supervised_results[subset_name]
                        acc = np.mean([v["acc"] for v in classifier_res.values()])
                        auc = np.mean([v["auc"] for v in classifier_res.values()])
                        if "ACC" in current_res:
                            current_res["ACC"] = acc
                        if "AUC" in current_res:
                            current_res["AUC"] = auc

                        if self.save_classifier_results:
                            for clf_name, clf_res in classifier_res.items():
                                if clf_name not in clf_rows:
                                    clf_rows[clf_name] = {
                                        "ACC": {"Dataset": ds_name},
                                        "AUC": {"Dataset": ds_name},
                                    }
                                clf_rows[clf_name]["ACC"][p] = clf_res["acc"]
                                clf_rows[clf_name]["AUC"][p] = clf_res["auc"]

                    if "model_agnostic" in self.eval_type:
                        try:
                            res_aad = AAD(pd.DataFrame(X), indices[:k])
                            if "AAD" in current_res:
                                current_res["AAD"] = res_aad
                        except Exception as e:
                            issues.add(f"AAD failed: {type(e).__name__}: {e}")

                    if "custom" in self.eval_type:
                        X_subset = X[:, indices[:k]]
                        for c_name, c_func in self.custom_metrics.items():
                            try:
                                val = c_func(X, X_subset, y)
                                if c_name in current_res:
                                    current_res[c_name] = val
                            except Exception as e:
                                issues.add(f"{c_name} failed: {type(e).__name__}: {e}")

                    for met in self.selected_metrics:
                        rows[met][p] = current_res[met]

                for met in self.selected_metrics:
                    ds_results[scale_name][met].append(rows[met])

                if self.save_classifier_results:
                    for clf_name, metric_dicts in clf_rows.items():
                        classifier_results_all[scale_name].setdefault(
                            clf_name, {"ACC": [], "AUC": []}
                        )
                        classifier_results_all[scale_name][clf_name]["ACC"].append(
                            metric_dicts["ACC"]
                        )
                        classifier_results_all[scale_name][clf_name]["AUC"].append(
                            metric_dicts["AUC"]
                        )

            # One repeat finished: advance this item's progress (one writer per key).
            if progress is not None:
                progress[key] = min(item_total, (r + 1) * units_per_repeat)

        stability = {}
        if self.stability_metrics:
            stability = self._compute_stability(ds_results, n_features, inner_jobs)

        return {
            "ds": ds_name,
            "method": name,
            "status": "ok",
            "elapsed": time.time() - t0,
            "n_samples": n_samples,
            "n_features": n_features,
            "repeats": repeats,
            "issues": sorted(issues),
            "ds_results": ds_results,
            "clf_results": classifier_results_all,
            "stability": stability,
        }

    # ------------------------------------------------------------------
    # Stability (computed in the worker, saved by the main process)
    # ------------------------------------------------------------------

    @staticmethod
    def _stability_step(ks, percentages, row):
        y_values = [row.get(p, np.nan) for p in percentages]
        if len(y_values) < 2 or any(np.isnan(yv) for yv in y_values):
            return np.nan
        try:
            _, df_der = fsdem.approx_func(ks, y_values)
            return fsdem.stability(df_der, start=min(ks), end=max(ks))
        except Exception:
            return np.nan

    def _compute_stability(self, ds_results, n_features, inner_jobs=1):
        """Mean stability of the metric curves, per (scale, metric)."""
        jobs, job_keys = [], []
        for scale_name, percentages in self.scales.items():
            ks = [
                max(1, min(math.ceil(p * n_features), n_features))
                for p in percentages
            ]
            for metric in self.stability_metrics:
                if metric not in ds_results[scale_name]:
                    continue
                for row in ds_results[scale_name][metric]:
                    jobs.append((ks, percentages, row))
                    job_keys.append((scale_name, metric))
        if not jobs:
            return {}

        vals = Parallel(n_jobs=max(1, min(inner_jobs, len(jobs))), backend="loky")(
            delayed(self._stability_step)(ks, percentages, row)
            for ks, percentages, row in jobs
        )
        grouped = defaultdict(list)
        for key, v in zip(job_keys, vals):
            grouped[key].append(v)
        return {key: (np.nanmean(v) if v else np.nan) for key, v in grouped.items()}

    # ------------------------------------------------------------------
    # Persistence (main process only)
    # ------------------------------------------------------------------

    @staticmethod
    def _read_csv_clean(fname):
        """Read a result CSV and merge duplicated header blocks left by older versions."""
        raw = pd.read_csv(fname, header=None, dtype=str)
        labels = raw.iloc[0].tolist()
        body = raw.iloc[1:].reset_index(drop=True)
        cols = {}
        for pos, lab in enumerate(labels):
            s = body.iloc[:, pos]
            cols[lab] = s if lab not in cols else cols[lab].combine_first(s)
        df = pd.DataFrame(cols)
        for c in df.columns:
            if c != "Dataset":
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def _order_rows(self, df):
        """Stable sort by position in the dataset list; unknown datasets go last."""
        n_unknown = len(self._ds_order)
        rank = df["Dataset"].map(lambda d: self._ds_order.get(d, n_unknown)).to_numpy()
        return df.iloc[np.argsort(rank, kind="stable")].reset_index(drop=True)

    @staticmethod
    def _atomic_write(df, fname):
        tmp = fname + ".tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, fname)

    def _upsert(self, fname, new_df, value_cols):
        """Replace the rows of new_df's datasets in fname, keep the rest, reorder, write."""
        cols = ["Dataset"] + list(value_cols)
        new_df = new_df.reindex(columns=cols)
        out = new_df
        if os.path.exists(fname):
            try:
                old_df = self._read_csv_clean(fname).reindex(columns=cols)
            except Exception as e:
                print(f"  WARNING: unreadable {fname} ({e}); moved to .corrupt")
                os.replace(fname, fname + ".corrupt")
                old_df = None
            if old_df is not None:
                old_df = old_df[~old_df["Dataset"].isin(new_df["Dataset"])]
                if not old_df.empty:
                    out = pd.concat([old_df, new_df], ignore_index=True)
        self._atomic_write(self._order_rows(out), fname)

    def _tidy_rows(self, rows, value_cols):
        df = pd.DataFrame(rows).rename(
            columns=lambda c: c if c == "Dataset" else self._col_label(c)
        )
        df = df.reindex(columns=["Dataset"] + value_cols)
        if not self.save_all:
            df = df.groupby("Dataset", as_index=False, sort=False)[value_cols].mean()
        return df

    def _save_results(self, method_name, ds_results):
        for scale_name, metric_data in ds_results.items():
            value_cols = self._pct_cols(scale_name)
            for metric, rows in metric_data.items():
                if not rows:
                    continue
                fname = self._metric_path(method_name, metric, scale_name)
                self._upsert(fname, self._tidy_rows(rows, value_cols), value_cols)

    def _save_classifier_results(self, method_name, ds_name, classifier_results_all):
        for scale_name, classifiers in classifier_results_all.items():
            value_cols = self._pct_cols(scale_name)
            for clf_name, metric_data in classifiers.items():
                for metric, rows in metric_data.items():
                    if not rows:
                        continue
                    fname = self._classifier_path(method_name, clf_name, metric, scale_name)
                    self._upsert(fname, self._tidy_rows(rows, value_cols), value_cols)

    def _save_stability(self, method_name, ds_name, stability):
        for (scale_name, metric), val in stability.items():
            fname = self._stability_path(method_name, metric, scale_name)
            row = pd.DataFrame([{"Dataset": ds_name, "Stability": val}])
            self._upsert(fname, row, ["Stability"])

    def _persist(self, res):
        self._save_results(res["method"], res["ds_results"])
        if self.save_classifier_results:
            self._save_classifier_results(res["method"], res["ds"], res["clf_results"])
        self._save_stability(res["method"], res["ds"], res["stability"])

    def _dump_unsaved(self, res):
        """Last resort: keep a computed result if writing the CSVs failed."""
        try:
            d = os.path.join(self.output_dir, "_unsaved")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f"{res['method']}_{res['ds']}.pkl"), "wb") as f:
                pickle.dump(res, f)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Post-hoc refinement
    # ------------------------------------------------------------------

    def refine(self, datasets=None, verbose=True):
        """
        Normalise every result CSV in output_dir:
          * merge duplicated header blocks left by older versions,
          * sort percentage columns numerically,
          * order rows like `datasets` (default: the list of the last run()).
        Datasets not in the list are kept and placed last. Timing files are untouched.
        """
        if datasets is not None:
            self._ds_order = {d: i for i, d in enumerate(datasets)}
        n_files = 0
        for fname in sorted(glob.glob(os.path.join(self.output_dir, "*.csv"))):
            if os.path.basename(fname).startswith("time_analysis"):
                continue
            try:
                df = self._read_csv_clean(fname)
            except Exception:
                continue
            if "Dataset" not in df.columns:
                continue
            value_cols = [c for c in df.columns if c != "Dataset"]
            try:
                value_cols.sort(key=float)
            except ValueError:
                pass  # e.g. Stability files
            df = self._order_rows(df[["Dataset"] + value_cols])
            self._atomic_write(df, fname)
            n_files += 1
        if verbose:
            print(f"  Refined {n_files} result files (headers, column and dataset order)")

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def run(self, datasets, methods, classifier=None, force=False):
        warnings.filterwarnings("ignore")
        global_start = time.time()

        datasets = list(datasets)
        self._ds_order = {d: i for i, d in enumerate(datasets)}
        manifest = self._load_manifest()
        fingerprints = {m["name"]: self._fingerprint(m) for m in methods}
        cache = {}

        work, skipped = [], []
        for ds_name in datasets:
            for m_info in methods:
                if force:
                    done, why = False, "forced"
                else:
                    done, why = self._method_is_complete(ds_name, m_info, manifest, cache)
                if done:
                    skipped.append((ds_name, m_info["name"], why))
                else:
                    work.append((ds_name, m_info))

        n_work = len(work)
        n_skip = len(skipped)
        n_outer, inner_jobs = self._allocate_cores(n_work)

        units_per_repeat = self._units_per_repeat()
        item_units = {}
        for ds_name, m_info in work:
            repeats = self.avg_steps if m_info.get("stochastic", False) else 1
            item_units[self._key(ds_name, m_info["name"])] = repeats * units_per_repeat
        total_units = sum(item_units.values())

        print()
        print("╔══════════════════════════════════════════════════════════╗")
        print("║                    FSEVAL  -  Running                    ║")
        print("╚══════════════════════════════════════════════════════════╝")
        print(f"  Datasets     : {len(datasets)}")
        print(f"  Methods      : {len(methods)}  ({', '.join(m['name'] for m in methods)})")
        print(f"  Metrics      : {', '.join(self.selected_metrics)}")
        print(f"  Scales       : {', '.join(self.scales.keys())}")
        print(f"  Work items   : {n_work} to run,  {n_skip} already done")
        print(f"  Logical CPUs : {self.n_cores}  "
              f"(outer={n_outer},  inner/worker={inner_jobs})")
        print(f"  Progress     : {total_units} steps  "
              f"({units_per_repeat} per method run)")
        print(f"  Output       : {self.output_dir}/")
        print("────────────────────────────────────────────────────────────")

        # Column widths for the per-item report lines.
        w_m = max([len(m["name"]) for m in methods] + [6])
        w_ds = max([len(str(d)) for d in datasets] + [7])

        for ds, m, why in skipped:
            print(f"  {'SKIP':<4} {m:<{w_m}}  {ds:<{w_ds}}  {why}")

        if n_work == 0:
            print("\n  Nothing left to do.")
            self.refine(datasets)
            print("════════════════════════════════════════════════════════════\n")
            return

        print()

        manager = Manager()
        progress = manager.dict()  # one writer per key, so no lock is needed

        if HAS_TQDM:
            bar = tqdm(
                total=total_units,
                desc="Progress",
                unit="step",
                bar_format=(
                    "  {l_bar}{bar}| {n_fmt}/{total_fmt} ({percentage:0.2f}%) "
                    "[{elapsed}<{remaining}, {rate_fmt}]"
                ),
                leave=True,
                mininterval=0.5,
            )
        else:
            bar = None
        last_print = [0]

        stop_poll = threading.Event()
        # Redraw often in a terminal; rarely when output goes to a log file.
        poll_interval = 0.5 if sys.stderr.isatty() else 30.0

        def _poll():
            while not stop_poll.is_set():
                try:
                    cur = sum(progress.values())
                except Exception:
                    break
                if bar is not None:
                    bar.n = cur
                    bar.refresh()
                elif cur - last_print[0] >= max(1, total_units // 50):
                    pct = 100.0 * cur / total_units if total_units else 0
                    print(
                        f"\r  Progress  {cur}/{total_units} ({pct:.2f}%)  "
                        f"elapsed {self._fmt_time(time.time() - global_start)}",
                        end="",
                        flush=True,
                    )
                    last_print[0] = cur
                stop_poll.wait(poll_interval)

        poller = threading.Thread(target=_poll, daemon=True)
        poller.start()

        def _report(res):
            status = res.get("status", "?")
            elapsed = self._fmt_time(res.get("elapsed", 0))
            total_so_far = self._fmt_time(time.time() - global_start)
            if status == "ok":
                shape = f"{res['n_samples']}x{res['n_features']}"
                msg = (
                    f"  {'OK':<4} {res['method']:<{w_m}}  {res['ds']:<{w_ds}}  "
                    f"{shape:<12}  {elapsed}  (total {total_so_far})"
                )
            else:
                detail = f" {res['detail']}" if res.get("detail") else ""
                msg = (
                    f"  {'FAIL':<4} {res['method']:<{w_m}}  {res['ds']:<{w_ds}}  "
                    f"{'':<12}  {elapsed}  ({status}){detail}"
                )
            for issue in res.get("issues", []):
                msg += f"\n       ! {issue}"
            if bar is not None:
                bar.write(msg)
            else:
                print()
                print(msg)

        def make_iter(mode):
            return Parallel(n_jobs=n_outer, backend="loky", return_as=mode)(
                delayed(self._run_one)(
                    ds, m, classifier, inner_jobs,
                    progress, self._key(ds, m["name"]), units_per_repeat,
                )
                for ds, m in work
            )

        try:
            try:
                results_iter = make_iter("generator_unordered")
            except (TypeError, ValueError):
                results_iter = make_iter("generator")  # older joblib

            for res in results_iter:
                if res.get("status") == "ok":
                    try:
                        self._persist(res)
                        manifest.setdefault(res["method"], {})[res["ds"]] = \
                            fingerprints[res["method"]]
                        self._save_manifest(manifest)
                    except Exception as e:
                        res["status"] = "save_failed"
                        res["detail"] = f"{type(e).__name__}: {e}"
                        self._dump_unsaved(res)
                # finished items count as fully done, even when they failed
                progress[self._key(res["ds"], res["method"])] = \
                    item_units.get(self._key(res["ds"], res["method"]), 0)
                _report(res)
        finally:
            stop_poll.set()
            poller.join(timeout=2.0)
            try:
                final = sum(progress.values())
            except Exception:
                final = total_units
            if bar is not None:
                bar.n = final
                bar.refresh()
                bar.close()
            else:
                print()
            manager.shutdown()

        print("────────────────────────────────────────────────────────────")
        self.refine(datasets)
        total_elapsed = time.time() - global_start
        print(f"  All done in {self._fmt_time(total_elapsed)}")
        print("════════════════════════════════════════════════════════════")
        print()

    # ------------------------------------------------------------------
    # Scalability timer
    # ------------------------------------------------------------------

    def timer(self, methods, vary_param="both", time_limit=3600):
        """
        Time each method on random data. Methods are called with their configured
        `params`; supervised methods receive a 5-class label vector.
        Note: time_limit is checked after a call returns; a running call is not interrupted.
        """
        experiments = []
        if vary_param in ["features", "both"]:
            experiments.append(
                {
                    "name": "features",
                    "fixed_val": 100,
                    "range": range(1000, 20001, 500),
                    "file": "time_analysis_features.csv",
                }
            )
        if vary_param in ["instances", "both"]:
            experiments.append(
                {
                    "name": "instances",
                    "fixed_val": 100,
                    "range": range(1000, 20001, 500),
                    "file": "time_analysis_instances.csv",
                }
            )

        for exp in experiments:
            vary_type = exp["name"]
            val_range = exp["range"]
            filename = os.path.join(self.output_dir, exp["file"])
            timed_out_methods = set()
            results = {m["name"]: [] for m in methods}

            for val in val_range:
                if vary_type == "features":
                    n_samples, n_features = exp["fixed_val"], val
                else:
                    n_samples, n_features = val, exp["fixed_val"]

                try:
                    X = np.random.rand(n_samples, n_features)
                except MemoryError:
                    for m in methods:
                        results[m["name"]].append(
                            -1 if m["name"] in timed_out_methods else np.nan
                        )
                    continue

                y_dummy = pd.Series(np.arange(n_samples) % 5)

                for m_info in methods:
                    name = m_info["name"]
                    fs_type = m_info["type"]
                    func = m_info["func"]
                    params = m_info.get("params", {})

                    if name in timed_out_methods:
                        results[name].append(-1)
                        continue

                    try:
                        start_time = time.time()
                        if fs_type == "unsupervised":
                            func(X, **params)
                        elif fs_type == "supervised":
                            func(X, y_dummy, **params)
                        else:
                            raise ValueError(
                                "Feature selection algorithm type is not valid."
                            )
                        duration = time.time() - start_time
                        if duration > time_limit:
                            timed_out_methods.add(name)
                        results[name].append(duration)
                    except Exception:
                        results[name].append(np.nan)

            df_results = pd.DataFrame.from_dict(
                results, orient="index", columns=list(val_range)
            )
            df_results.index.name = "Method"
            self._atomic_write(df_results.reset_index(), filename)
            print(f"Saved timing results to {filename}")
