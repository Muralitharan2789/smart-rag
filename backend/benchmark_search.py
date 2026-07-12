import time
import numpy as np
from search import vector_search


def random_query_vector(dim=384):
    v = np.random.normal(size=dim)
    return (v / np.linalg.norm(v)).tolist()


def benchmark(n_trials=10):
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        vector_search(random_query_vector(), top_k=10)
        times.append((time.perf_counter() - t0) * 1000)
    print(f"Avg: {sum(times)/len(times):.2f} ms  |  Min: {min(times):.2f} ms  |  Max: {max(times):.2f} ms  ({n_trials} trials)")


if __name__ == "__main__":
    benchmark()