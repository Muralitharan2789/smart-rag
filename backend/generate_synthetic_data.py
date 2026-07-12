import random
import numpy as np
from database import get_connection

SYNTHETIC_DOC_NAME = "synthetic_scale_test.txt"
NUM_CHUNKS = 5000

WORDS = [
    "revenue", "quarterly", "report", "synergy", "alpha", "beta", "dataset",
    "cluster", "vector", "north", "south", "asia", "europe", "market",
    "growth", "decline", "forecast", "budget", "overview", "summary",
]


def random_text(n_words=20):
    return " ".join(random.choice(WORDS) for _ in range(n_words))


def random_embedding(dim=384):
    # Random, semantically meaningless vectors — fine for measuring SPEED,
    # not valid for measuring retrieval QUALITY. That distinction matters:
    # today tests "how fast," not "how accurate."
    v = np.random.normal(size=dim)
    return (v / np.linalg.norm(v)).tolist()


def generate():
    conn = get_connection()
    cur = conn.cursor()
    print(f"Inserting {NUM_CHUNKS} synthetic chunks...")
    for i in range(NUM_CHUNKS):
        cur.execute(
            "INSERT INTO chunks (document_name, chunk_text, chunk_type, embedding) VALUES (%s, %s, %s, %s)",
            (SYNTHETIC_DOC_NAME, random_text(), "text", random_embedding()),
        )
        if (i + 1) % 500 == 0:
            conn.commit()
            print(f"  {i + 1}/{NUM_CHUNKS} inserted...")
    conn.commit()
    conn.close()
    print("Done.")


if __name__ == "__main__":
    generate()