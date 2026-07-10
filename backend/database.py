import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    return psycopg2.connect(os.getenv("DATABASE_URL"))

def check_db_connection() -> bool:
    try:
        conn = get_connection()
        conn.close()
        return True
    except Exception as e:
        print(f"DB connection failed: {e}")
        return False


def insert_chunk(document_name: str, chunk_text: str, chunk_type: str, embedding: list[float]) -> int:
    """Inserts one chunk with its embedding, returns the new row's id."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO chunks (document_name, chunk_text, chunk_type, embedding)
            VALUES (%s, %s, %s, %s)
            RETURNING id;
            """,
            (document_name, chunk_text, chunk_type, embedding),
        )
        new_id = cur.fetchone()[0]
        conn.commit()
        return new_id
    finally:
        conn.close()


def count_chunks(document_name: str | None = None) -> int:
    """Returns how many chunks are stored, optionally filtered to one document."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        if document_name:
            cur.execute("SELECT COUNT(*) FROM chunks WHERE document_name = %s;", (document_name,))
        else:
            cur.execute("SELECT COUNT(*) FROM chunks;")
        return cur.fetchone()[0]
    finally:
        conn.close()