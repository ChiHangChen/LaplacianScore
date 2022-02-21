
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


def cal_lap_score(features, D, L):
    features_ = features - np.sum((features @ D) / np.sum(D))
    L_score = (features_ @ L @ features_) / (features_ @ D @ features_)
    return L_score


def get_k_nearest(dist, k, sample_index):
    # dist is zero means it is the sample itself
    return sorted(
        range(len(dist)),
        key=lambda i: dist[i] if i != sample_index else np.inf
    )[:k] + [sample_index]


def laplacian_score(df_arr, label=None, **kwargs):
    kwargs.setdefault("k_nearest", 8)

    distances = pdist(df_arr, metric='euclidean')
    dist_matrix = squareform(distances)
    del distances
    edge_sparse_matrix = pd.DataFrame(
        np.zeros((df_arr.shape[0], df_arr.shape[0])),
        dtype=int
    )
    if label is None:
        for row_id, row in enumerate(dist_matrix):
            k_nearest_id = get_k_nearest(row, kwargs["k_nearest"], row_id)
            edge_sparse_matrix.iloc[k_nearest_id, k_nearest_id] = 1
    else:
        label = np.array(label)
        unique_label = np.unique(label)
        for i in unique_label:
            group_index = np.where(label == i)[0]
            edge_sparse_matrix.iloc[group_index, group_index] = 1
    S = dist_matrix * edge_sparse_matrix
    del dist_matrix, edge_sparse_matrix
    D = np.diag(S.sum(axis=1))
    L = D - S
    del S
    features_lap_score = np.apply_along_axis(
        func1d=lambda f: cal_lap_score(f, D, L), axis=0, arr=df_arr
    )
    return features_lap_score
