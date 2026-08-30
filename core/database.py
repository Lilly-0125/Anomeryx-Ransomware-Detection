import os
import sys
import sqlite3


class AlertDatabase:
    def __init__(self):
        self.db_path = self._get_database_path()
        self._create_table()

    def _get_database_path(self):
        # Running as Anomeryx.exe
        if getattr(sys, "frozen", False):
            base_dir = os.path.dirname(sys.executable)

        # Running using python main.py
        else:
            base_dir = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )

        database_dir = os.path.join(base_dir, "database")
        os.makedirs(database_dir, exist_ok=True)

        return os.path.join(database_dir, "anomeryx.db")
    
    def connect(self):
        return sqlite3.connect(self.db_path)

    def _create_table(self):
        with self.connect() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    score REAL NOT NULL,
                    confidence REAL NOT NULL,
                    details TEXT
                )
            """)

            conn.commit()

    def add_alert(
        self,
        timestamp,
        level,
        score,
        confidence,
        details
    ):
        with self.connect() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO alerts (
                    timestamp,
                    level,
                    score,
                    confidence,
                    details
                )
                VALUES (?, ?, ?, ?, ?)
            """, (
                timestamp,
                level,
                score,
                confidence,
                details
            ))

            conn.commit()

    def get_all_alerts(self):
        with self.connect() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    id,
                    timestamp,
                    level,
                    score,
                    confidence,
                    details
                FROM alerts
                ORDER BY id ASC
            """)

            return cursor.fetchall()

    def clear_alerts(self):
        with self.connect() as conn:
            cursor = conn.cursor()

            cursor.execute("DELETE FROM alerts")
            conn.commit()