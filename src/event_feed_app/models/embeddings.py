
from sentence_transformers import SentenceTransformer
import os
from typing import Dict, List, Tuple, Iterable, Optional
import numpy as np
def get_embedding_model(name: str, max_seq_tokens: int) -> SentenceTransformer:
    model = SentenceTransformer(name)
    limit = getattr(model, "max_seq_length", 512) or 512
    model.max_seq_length = max(32, min(max_seq_tokens, int(limit)))
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    return model

def encode_texts(model: SentenceTransformer, texts: List[str], batch_size: int) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
