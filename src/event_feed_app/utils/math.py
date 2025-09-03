from scipy import sparse
import numpy as np

def softmax_rows(S, T: float = 1.0) -> np.ndarray:
    if sparse.issparse(S):
        S = S.toarray()
    else:
        S = np.asarray(S)
    S = S.astype(np.float64, copy=False)
    S = S / max(T, 1e-6)
    S -= np.max(S, axis=1, keepdims=True)
    S = np.exp(S)
    denom = np.sum(S, axis=1, keepdims=True) + 1e-12
    return S / denom

def entropy_rows(P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    P_clip = np.clip(P, eps, 1.0)
    return -np.sum(P_clip * np.log(P_clip), axis=1)

def topk_sum(P: np.ndarray, k: int = 3) -> np.ndarray:
    if P.shape[1] <= k:
        return np.sum(P, axis=1)
    part = np.partition(P, -k, axis=1)
    return np.sum(part[:, -k:], axis=1)
