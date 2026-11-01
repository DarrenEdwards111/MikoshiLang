"""Machine-learning wrappers around scikit-learn (optional dependency).

Results are association-style rule lists.  Legacy versions returned Error[...]
expressions when scikit-learn was missing and accepted ragged data; these
validate input and raise instead."""
import sympy as sp

from ..expr import Expr, Symbol
from .core import Unevaluated, fn, error, record, S, R, L, rule, list_arg, num_list, pos_int, is_list

AREA = "ml"


def _sk(mod):
    try:
        return __import__(mod, fromlist=["_"])
    except ImportError:
        raise NotImplementedError("this function requires scikit-learn (pip install mikoshilang[all])")


def _matrix(x, fname):
    rows = list_arg(x, fname, "data", 1)
    if not all(is_list(r) for r in rows):
        raise ValueError(f"{fname}: data must be a list of numeric feature vectors")
    out = [[float(v) for v in num_list(r, fname, "feature vector")] for r in rows]
    if len({len(r) for r in out}) != 1:
        raise ValueError(f"{fname}: all feature vectors must have the same length")
    return out


def _labels(y, n, fname):
    ys = [float(v) for v in num_list(y, fname, "labels")]
    if len(ys) != n:
        raise ValueError(f"{fname}: {n} samples but {len(ys)} labels")
    if len(set(ys)) < 2:
        raise ValueError(f"{fname}: labels must contain at least two classes")
    return ys


def _s(name):
    return Symbol(name)


@fn("KMeans", area=AREA, status="verified",
    notes="KMeans[data, k, maxIter=100] -> {labels -> ..., centers -> ...} (scikit-learn, random_state=42, "
          "n_init=10); k must not exceed the number of points"
          ". Check: reference (differential cases in tests/library).")
def _kmeans(data, k, maxit=100):
    X = _matrix(data, "KMeans")
    k = pos_int(k, "KMeans", "k")
    if k > len(X):
        raise ValueError("KMeans: k exceeds the number of data points")
    cl = _sk("sklearn.cluster")
    m = cl.KMeans(n_clusters=k, max_iter=pos_int(maxit, "KMeans", "maxIter"), random_state=42, n_init=10)
    labels = m.fit_predict(X)
    return L(rule(_s("labels"), R(labels)), rule(_s("centers"), R(m.cluster_centers_)))


@fn("PCA", area=AREA, status="verified",
    notes="PCA[data, n=2] -> {transformed -> ..., explainedVariance -> ratios} (scikit-learn; component signs "
          "are arbitrary)"
          ". Check: reference (differential cases in tests/library).")
def _pca(data, n=2):
    X = _matrix(data, "PCA")
    n = pos_int(n, "PCA", "number of components")
    if n > min(len(X), len(X[0])):
        raise ValueError("PCA: n exceeds min(samples, features)")
    dec = _sk("sklearn.decomposition")
    p = dec.PCA(n_components=n)
    t = p.fit_transform(X)
    return L(rule(_s("transformed"), R(t)), rule(_s("explainedVariance"), R(p.explained_variance_ratio_)))


@fn("LogisticRegressionTrain", area=AREA, status="verified",
    notes="-> {coefficients, intercept, accuracy (training set)}; scikit-learn LogisticRegression (L2, C=1); "
          "binary labels only"
          ". Check: reference (differential cases in tests/library).")
def _logreg(X, y):
    Xs = _matrix(X, "LogisticRegressionTrain")
    ys = _labels(y, len(Xs), "LogisticRegressionTrain")
    if len(set(ys)) != 2:
        raise ValueError("LogisticRegressionTrain: only binary classification is supported")
    lm = _sk("sklearn.linear_model")
    m = lm.LogisticRegression(max_iter=1000)
    m.fit(Xs, ys)
    return L(rule(_s("coefficients"), R(m.coef_[0])), rule(_s("intercept"), float(m.intercept_[0])),
             rule(_s("accuracy"), float(m.score(Xs, ys))))


@fn("DecisionTreeTrain", area=AREA, status="verified",
    notes="-> {accuracy (training set), depth}; scikit-learn DecisionTreeClassifier(max_depth, random_state=42)"
          ". Check: reference (differential cases in tests/library).")
def _tree(X, y, depth=5):
    Xs = _matrix(X, "DecisionTreeTrain")
    ys = _labels(y, len(Xs), "DecisionTreeTrain")
    tr = _sk("sklearn.tree")
    m = tr.DecisionTreeClassifier(max_depth=pos_int(depth, "DecisionTreeTrain", "max depth"), random_state=42)
    m.fit(Xs, ys)
    return L(rule(_s("accuracy"), float(m.score(Xs, ys))), rule(_s("depth"), int(m.get_depth())))
