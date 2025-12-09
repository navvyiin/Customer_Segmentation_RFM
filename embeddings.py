import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

try:
    import faiss  # type: ignore

    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False

try:
    from annoy import AnnoyIndex

    ANNOY_AVAILABLE = True
except ImportError:
    AnnoyIndex = None
    ANNOY_AVAILABLE = False


def build_customer_product_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a customer × product matrix from transactions.
    Expects columns: CustomerID, StockCode, Quantity.
    """
    if "StockCode" not in df.columns:
        raise ValueError("DataFrame must contain 'StockCode' for embeddings.")

    d = df[["CustomerID", "StockCode", "Quantity"]].copy()
    mat = (
        d.groupby(["CustomerID", "StockCode"])["Quantity"]
        .sum()
        .unstack(fill_value=0)
    )
    return mat


def compute_svd_embeddings(
    cust_prod_mat: pd.DataFrame,
    n_components: int = 64,
):
    """
    Compute low-dimensional customer embeddings using TruncatedSVD.
    """
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    emb = svd.fit_transform(cust_prod_mat.values)

    cust_ids = cust_prod_mat.index.astype(str).tolist()
    product_cols = cust_prod_mat.columns.astype(str).tolist()
    return emb, cust_ids, product_cols


def build_similarity_index(embeddings: np.ndarray, metric: str = "angular"):
    """
    Build a similarity index using Faiss or Annoy if available.
    Fallback: return embeddings with 'bruteforce' mode.
    """
    n, d = embeddings.shape

    if FAISS_AVAILABLE:
        index = faiss.IndexFlatL2(d)
        index.add(embeddings.astype("float32"))
        return index, "faiss"

    if ANNOY_AVAILABLE:
        index = AnnoyIndex(d, metric)
        for i, vec in enumerate(embeddings):
            index.add_item(i, vec.tolist())
        index.build(10)
        return index, "annoy"

    # Fallback: brute-force cosine similarity
    return embeddings, "bruteforce"


def query_similarity(index, all_embeddings, query_vec, top_k: int = 10, kind: str = "faiss"):
    """
    Query top-K similar customers given a index kind:
        - 'faiss'
        - 'annoy'
        - 'bruteforce'
    """
    if kind == "faiss":
        q = np.array(query_vec, dtype="float32").reshape(1, -1)
        dists, idxs = index.search(q, top_k)
        return idxs[0].tolist(), dists[0].tolist()

    elif kind == "annoy":
        idxs, dists = index.get_nns_by_vector(
            query_vec.tolist(), top_k, include_distances=True
        )
        return idxs, dists

    else:  # brute-force cosine similarity
        a = all_embeddings
        q = np.array(query_vec, dtype=float)
        q_norm = q / (np.linalg.norm(q) + 1e-9)

        norms = np.linalg.norm(a, axis=1, keepdims=True)
        a_norm = a / np.clip(norms, 1e-9, None)
        sims = a_norm @ q_norm
        idxs = np.argsort(-sims)[:top_k]
        dists = (1 - sims[idxs]).tolist()
        return idxs.tolist(), dists