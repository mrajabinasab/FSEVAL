import numpy as np
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics.cluster import normalized_mutual_info_score
from clustpy.metrics.clustering_metrics import unsupervised_clustering_accuracy
from sklearn.cluster import KMeans
from statistics import mean

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.linear_model._logistic")


def unsupervised_eval(data, labels, avg_steps=10):
    k = len(np.unique(labels))
    lc = LabelEncoder()
    # Handle both Series and arrays
    y_clean = labels.to_numpy().ravel() if hasattr(labels, 'to_numpy') else labels.ravel()
    labels_encoded = lc.fit_transform(y_clean)
    
    clsacc = np.empty(avg_steps)
    nmi = np.empty(avg_steps)
    for i in range(avg_steps):
        kmeans = KMeans(n_clusters=k, random_state=i, n_init="auto").fit(data)
        clsacc[i] = unsupervised_clustering_accuracy(labels_encoded, kmeans.labels_)
        nmi[i] = normalized_mutual_info_score(labels_encoded, kmeans.labels_)
    return clsacc.mean(), nmi.mean()


def supervised_eval(X, y, classifier=None, cv=5, avg_steps=10):
    """
    Evaluates using a provided sklearn classifier instance. 
    Defaults to RandomForest if None.
    """
    if classifier is None:
        classifier = RandomForestClassifier(n_estimators=100)
        
    result_auc = []
    result_acc = []
    
    lc = LabelEncoder()
    y_clean = y.to_numpy().ravel() if hasattr(y, 'to_numpy') else y.ravel()
    y_encoded = lc.fit_transform(y_clean)
    
    auc_scoring = 'roc_auc_ovr' if len(np.unique(y_encoded)) > 2 else 'roc_auc'
    scoring_metrics = {'auc': auc_scoring, 'acc': 'accuracy'}
    
    for i in range(avg_steps):
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=i)
        
        # Ensure repeatability if the classifier supports random_state
        if hasattr(classifier, 'random_state'):
            classifier.random_state = i
            
        scores = cross_validate(classifier, X, y_encoded, scoring=scoring_metrics, cv=skf)
        result_acc.append(scores['test_acc'].mean())
        result_auc.append(scores['test_auc'].mean())
                
    return mean(result_acc), mean(result_auc)