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


def rowwise_standardize(S, eps=1e-8, clip=8.0):
    S = np.asarray(S, dtype=np.float64)
    mu = S.mean(axis=1, keepdims=True)
    sd = S.std(axis=1, keepdims=True)
    Z = (S - mu) / (sd + eps)
    if clip is not None:
        Z = np.clip(Z, -clip, clip)
    return Z

def entropy_rows(P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    P_clip = np.clip(P, eps, 1.0)
    return -np.sum(P_clip * np.log(P_clip), axis=1)

def topk_sum(P: np.ndarray, k: int = 3) -> np.ndarray:
    if P.shape[1] <= k:
        return np.sum(P, axis=1)
    part = np.partition(P, -k, axis=1)
    return np.sum(part[:, -k:], axis=1)

import numpy as np
from scipy import sparse
try:
    from sklearn.preprocessing import normalize as sk_normalize
except Exception:
    sk_normalize = None

def l2_normalize_rows(A, eps: float = 1e-9, copy: bool = True):
    """
    Row-wise L2 normalize dense or sparse matrices. Zero/near-zero rows -> all zeros.
    - Dense path keeps dtype (e.g., float32) and can operate in-place.
    - Sparse path delegates to sklearn (CSR/CSC friendly).
    """
    if sparse.issparse(A):
        if sk_normalize is None:
            raise ImportError("scikit-learn is required for sparse normalization")
        return sk_normalize(A, norm="l2", copy=copy)

    A = np.asarray(A)
    if copy:
        A = A.copy()

    norms = np.linalg.norm(A, axis=1, keepdims=True)          # (N,1)
    row_mask = (norms[:, 0] > eps)                            # (N,)

    # divide rows with sufficient norm (broadcasts (k,1) across D)
    A[row_mask] /= norms[row_mask]

    # force ill-conditioned rows to exact zeros
    A[~row_mask] = 0.0
    return A



# --- geometry fixes ---
def fit_abtt(Z: np.ndarray, k: int = 2):
    """
    Fit 'all-but-the-top' on Z (N,d). Returns mean and top-k PC directions.
    Choose k in [1..3]; tune on a dev set.
    """
    mu = Z.mean(axis=0, keepdims=True)
    Zc = Z - mu
    # principal axes live in Vt (d x d); take first k columns
    # SVD on data matrix is stable and fast for d<<N
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    P = Vt[:k].T  # (d,k)
    return mu, P

def apply_abtt(A: np.ndarray, mu: np.ndarray, P: np.ndarray):
    Ac = A - mu
    return Ac - (Ac @ P) @ P.T