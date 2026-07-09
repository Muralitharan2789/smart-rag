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