
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import Normalizer
import pandas as pd
def pick_diverse_subset(df: pd.DataFrame, text_col: str, k_clusters: int, per_cluster: int, random_state: int = 42) -> pd.DataFrame:
    texts = df[text_col].fillna("").astype(str).tolist()
    hv = HashingVectorizer(analyzer="char", ngram_range=(3,5), n_features=2**20,
                           alternate_sign=False, strip_accents="unicode")
    X_hash = hv.transform(texts)
    svd_sketch = TruncatedSVD(n_components=96, random_state=random_state)
    X_sketch = svd_sketch.fit_transform(X_hash)
    X_sketch = Normalizer(copy=False).fit_transform(X_sketch)

    mbk = MiniBatchKMeans(n_clusters=k_clusters, random_state=random_state, batch_size=4096)
    labels = mbk.fit_predict(X_sketch)

    subset_idx = (pd.Series(range(len(texts)))
                  .groupby(labels, group_keys=False)
                  .apply(lambda s: s.sample(min(per_cluster, len(s)), random_state=random_state))
                  .tolist())
    return df.iloc[subset_idx].reset_index(drop=True)