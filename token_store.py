import sqlite3
import datetime
import os

class TokenStore:
    def __init__(self, db_path: str | None = None):
        # Default to 'tokens.db' or override via env var
        self.db_path = db_path or os.getenv("REPLYFLOW_TOKEN_STORE", "tokens.db")
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tokens (
                    user_id TEXT,
                    provider TEXT,
                    page_id TEXT,
                    page_access_token TEXT,
                    ig_account_id TEXT,
                    user_access_token TEXT,
                    created_at TEXT,
                    PRIMARY KEY (user_id, provider, page_id)
                )
            """)

    def save_token(self, user_id: str, provider: str,
                   page_id: str, page_access_token: str,
                   ig_account_id: str | None = None,
                   user_access_token: str | None = None) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO tokens
                (user_id, provider, page_id, page_access_token,
                 ig_account_id, user_access_token, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id, provider, page_id, page_access_token,
                ig_account_id, user_access_token,
                datetime.datetime.utcnow().isoformat()
            ))

    def get_page_token(self, user_id: str, provider: str) -> str | None:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("""
                SELECT page_access_token
                FROM tokens
                WHERE user_id = ? AND provider = ?
                LIMIT 1
            """, (user_id, provider))
            row = cur.fetchone()
            return row[0] if row else None

    def list_tokens(self, user_id: str, provider: str) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("""
                SELECT page_id, page_access_token,
                       ig_account_id, user_access_token, created_at
                FROM tokens
                WHERE user_id = ? AND provider = ?
            """, (user_id, provider))
            return [
                {
                    "page_id": row[0],
                    "page_access_token": row[1],
                    "ig_account_id": row[2],
                    "user_access_token": row[3],
                    "created_at": row[4],
                }
                for row in cur.fetchall()
            ]
