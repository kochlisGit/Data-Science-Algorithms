from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Executes Isolation Forest Algorithm to detect outliers.
def isolation_forest(data, n_estimators, contamination, standard_scaling):
    if standard_scaling:
        data = StandardScaler().fit_transform(data)

    return IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
        warm_start=True
    ).fit_predict(data)


# Executes Local Outlier Factor Algorithm to detect outliers.
def local_outlier_factor(data, n_neighbors, contamination, p, standard_scaling):
    if standard_scaling:
        data = StandardScaler().fit_transform(data)

    return LocalOutlierFactor(
        n_neighbors=n_neighbors,
        algorithm='auto',
        metric='euclidean',
        p=p,
        contamination=contamination,
        novelty=False,
        n_jobs=-1
    ).fit_predict(data)


# Executes Elliptic Envelope Algorithm to detect outliers.
def elliptic_envelope(data, support_fraction, contamination, standard_scaling):
    if standard_scaling:
        data = StandardScaler().fit_transform(data)

    return EllipticEnvelope(
        support_fraction=support_fraction,
        contamination=contamination,
        random_state=42,
    ).fit_predict(data)


# Executes DBSCAN Algorithm to detect outliers.
def dbscan(data, epsilon, min_samples, standard_scaling):
    if standard_scaling:
        data = StandardScaler().fit_transform(data)

    return DBSCAN(eps=epsilon,
                  min_samples=min_samples,
                  metric='euclidean',
                  algorithm='auto',
                  n_jobs=-1).fit_predict(data)


# Executes PCA DBSCAN Algorithm to detect outliers.
def pca_dbscan(data, n_components, epsilon, min_samples):
    data = PCA(n_components=n_components, whiten=True, random_state=42).fit_transform(data)

    return DBSCAN(eps=epsilon,
                  min_samples=min_samples,
                  metric='euclidean',
                  algorithm='auto',
                  n_jobs=-1).fit_predict(data)


# Constructs original data plot.
def plot_original_data(x, y, title, x_label, y_label):
    _, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.scatter(x=x, y=y)
    plt.show()


# Constructs outlier detection plot.
def plot_normal_outliers_data(orig_x, orig_y, out_x, out_y, title, x_label, y_label):
    _, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.scatter(x=orig_x, y=orig_y, color='blue')
    plt.scatter(x=out_x, y=out_y, color='red')
    plt.show()
