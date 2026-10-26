import numpy as np
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics.cluster import normalized_mutual_info_score, contingency_matrix
from sklearn.base import clone
from xgboost import XGBClassifier
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.linear_model._logistic")


# ---------------------------------------------------------------------------
# Unsupervised
# ---------------------------------------------------------------------------

def clustering_accuracy(labels_true, labels_pred):
    """
    Unsupervised clustering accuracy: the fraction of points correctly labelled
    under the best one-to-one matching between predicted clusters and classes,
    found with the Hungarian method. Equivalent to ClustPy's
    unsupervised_clustering_accuracy (zero padding to a square matrix does not
    change the optimal assignment value).

    References
    ----------
    Yang, Yi, et al. "Image clustering using local discriminant models and global
    integration." IEEE Transactions on Image Processing 19.10 (2010): 2761-2773.
    """
    cm = contingency_matrix(labels_true, labels_pred)
    rows, cols = linear_sum_assignment(cm, maximize=True)
    return cm[rows, cols].sum() / cm.sum()


def _unsupervised_subset_job(subset_name, data, labels_encoded, k, avg_steps, base_seed):
    """One job = one subset, all seeds. Coarser grain → better CPU utilisation."""
    clsacc_list, nmi_list = [], []
    for seed in range(avg_steps):
        kmeans = KMeans(
            n_clusters=k, random_state=base_seed + seed, n_init="auto"
        ).fit(data)
        clsacc_list.append(
            clustering_accuracy(labels_encoded, kmeans.labels_)
        )
        nmi_list.append(
            normalized_mutual_info_score(labels_encoded, kmeans.labels_)
        )
    return subset_name, float(np.mean(clsacc_list)), float(np.mean(nmi_list))


def unsupervised_eval_batch(X_subsets, y, avg_steps=10, n_jobs=-1, random_state=42):
    k = len(np.unique(y))
    y_clean = y.to_numpy().ravel() if hasattr(y, "to_numpy") else y.ravel()
    labels_encoded = LabelEncoder().fit_transform(y_clean)

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_unsupervised_subset_job)(
            subset_name, X, labels_encoded, k, avg_steps, random_state
        )
        for subset_name, X in X_subsets.items()
    )

    output = {}
    for subset_name, clsacc, nmi in results:
        output[subset_name] = {"clsacc": clsacc, "nmi": nmi}
    return output


# ---------------------------------------------------------------------------
# Supervised
# ---------------------------------------------------------------------------

def _get_default_classifiers():
    return {
        # Wrapped in CalibratedClassifierCV for multiclass roc_auc_ovr (see supervised_eval_batch)
        "SVM": SVC(kernel="rbf", C=1.0, gamma="scale"),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "XGBOOST": XGBClassifier(n_estimators=100, eval_metric="logloss", n_jobs=1),
        "RANDOMFOREST": RandomForestClassifier(n_estimators=100, n_jobs=1),
        "DECISIONTREE": DecisionTreeClassifier(),
    }


def _supervised_subset_clf_job(
    subset_name, clf_name, X, y_encoded, classifier, cv, avg_steps, base_seed, scoring_metrics
):
    """One job = one (subset, classifier), all seeds. Coarser grain → better CPU utilisation."""
    acc_list, auc_list = [], []
    for seed in range(avg_steps):
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=base_seed + seed)
        clf = clone(classifier)
        if hasattr(clf, "random_state"):
            clf.random_state = base_seed + seed
        scores = cross_validate(
            clf, X, y_encoded, scoring=scoring_metrics, cv=skf, n_jobs=1
        )
        acc_list.append(scores["test_acc"].mean())
        auc_list.append(scores["test_auc"].mean())
    return subset_name, clf_name, float(np.mean(acc_list)), float(np.mean(auc_list))


def supervised_eval_batch(
    X_subsets, y, classifier=None, cv=5, avg_steps=10, n_jobs=-1, random_state=42
):
    y_clean = y.to_numpy().ravel() if hasattr(y, "to_numpy") else y.ravel()
    y_encoded = LabelEncoder().fit_transform(y_clean)
    multiclass = len(np.unique(y_encoded)) > 2
    auc_scoring = "roc_auc_ovr" if multiclass else "roc_auc"
    scoring_metrics = {"auc": auc_scoring, "acc": "accuracy"}

    classifiers = (
        _get_default_classifiers()
        if classifier is None
        else {classifier.__class__.__name__: classifier}
    )
    # roc_auc_ovr needs predict_proba; SVC(probability=True) is deprecated since sklearn 1.9.
    if multiclass:
        classifiers = {
            name: CalibratedClassifierCV(clf, ensemble=False) if isinstance(clf, SVC) else clf
            for name, clf in classifiers.items()
        }

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_supervised_subset_clf_job)(
            subset_name,
            clf_name,
            X,
            y_encoded,
            clf,
            cv,
            avg_steps,
            random_state,
            scoring_metrics,
        )
        for subset_name, X in X_subsets.items()
        for clf_name, clf in classifiers.items()
    )

    output = {
        subset_name: {clf_name: {"acc": 0.0, "auc": 0.0} for clf_name in classifiers}
        for subset_name in X_subsets
    }
    for subset_name, clf_name, acc, auc in results:
        output[subset_name][clf_name]["acc"] = acc
        output[subset_name][clf_name]["auc"] = auc
    return output
