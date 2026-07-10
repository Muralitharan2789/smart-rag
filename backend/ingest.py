import sys
from pathlib import Path

from parser import parse_document
from chunker import chunk_document
from embedder import embed_batch
from database import insert_chunk, count_chunks


def ingest_document(file_path: str) -> None:
    document_name = Path(file_path).name
    print(f"Parsing {document_name}...")
    text = parse_document(file_path)

    print("Chunking...")
    chunks = chunk_document(text, max_chunk_size=800, overlap=100)
    print(f"  {len(chunks)} chunks created "
          f"({sum(1 for c in chunks if c.chunk_type == 'table')} tables, "
          f"{sum(1 for c in chunks if c.chunk_type == 'text')} text)")

    print("Embedding (batched)...")
    texts = [c.text for c in chunks]
    embeddings = embed_batch(texts)

    print("Storing in Postgres...")
    for chunk, embedding in zip(chunks, embeddings):
        insert_chunk(document_name, chunk.text, chunk.chunk_type, embedding)

    total = count_chunks(document_name)
    print(f"\nDone. {total} chunks now stored for '{document_name}'.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ingest.py <path-to-file>")
        sys.exit(1)
    ingest_document(sys.argv[1])