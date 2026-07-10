from sentence_transformers import SentenceTransformer

_model = None


def get_model() -> SentenceTransformer:
    """Lazy-loads the embedding model once and reuses it (loading takes a few seconds)."""
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return _model


def embed_text(text: str) -> list[float]:
    """Returns a 384-dimensional embedding for a single piece of text."""
    model = get_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """More efficient than calling embed_text in a loop — batches the model call."""
    model = get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embeddings.tolist()