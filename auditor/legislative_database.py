"""
Legislative Database Lookup Helper
==================================
Manages historical municipal records (ordinances, resolutions, zoning plans)
using a lightweight sqlite3 storage engine. Detects references in text and fetches records.
"""

import re
import sqlite3
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field


class LegislativeRecord(BaseModel):
    """
    Data model representing a historical municipal legislative record.
    """
    legislative_id: str = Field(..., description="Unique identifier (e.g. ORD-2025-012, RES-2026-004).")
    document_type: str = Field(..., description="Document category (e.g., Ordinance, Resolution, Zoning Plan).")
    title: str = Field(..., description="Brief title describing the legislative action.")
    summary: str = Field(..., description="Key provisions, summaries, or descriptions of the record.")
    passed_date: str = Field(..., description="ISO 8601 date string when the record was approved (YYYY-MM-DD).")
    status: str = Field("Passed", description="Operational status: Passed, Rescinded, Amended.")


class LegislativeDatabaseHelper:
    """
    SQLite-based helper to store, index, extract, and retrieve municipal actions.
    """

    def __init__(self, db_path: Path):
        """
        Initializes connection to SQLite database and ensures schema exists.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        """
        Ensures the legislative records table is configured in the database.
        """
        create_sql = """
        CREATE TABLE IF NOT EXISTS legislative_records (
            legislative_id TEXT PRIMARY KEY,
            document_type TEXT NOT NULL,
            title TEXT NOT NULL,
            summary TEXT NOT NULL,
            passed_date TEXT NOT NULL,
            status TEXT NOT NULL
        );
        """
        with self._conn:
            self._conn.execute(create_sql)

    def close(self) -> None:
        """
        Closes the active database connection.
        """
        self._conn.close()

    def upsert_record(self, record: LegislativeRecord) -> None:
        """
        Inserts a new record or replaces it if the ID already exists.
        """
        upsert_sql = """
        INSERT OR REPLACE INTO legislative_records
        (legislative_id, document_type, title, summary, passed_date, status)
        VALUES (?, ?, ?, ?, ?, ?);
        """
        with self._conn:
            self._conn.execute(
                upsert_sql,
                (
                    record.legislative_id.upper().strip(),
                    record.document_type.strip(),
                    record.title.strip(),
                    record.summary.strip(),
                    record.passed_date.strip(),
                    record.status.strip()
                )
            )

    def get_record(self, legislative_id: str) -> Optional[LegislativeRecord]:
        """
        Retrieves a single legislative record from the database by its ID.
        """
        select_sql = """
        SELECT legislative_id, document_type, title, summary, passed_date, status
        FROM legislative_records
        WHERE legislative_id = ?;
        """
        cursor = self._conn.execute(select_sql, (legislative_id.upper().strip(),))
        row = cursor.fetchone()
        
        if row:
            return LegislativeRecord(
                legislative_id=row["legislative_id"],
                document_type=row["document_type"],
                title=row["title"],
                summary=row["summary"],
                passed_date=row["passed_date"],
                status=row["status"]
            )
        return None

    def extract_ids_from_text(self, text: str) -> List[str]:
        """
        Applies regular expression patterns to extract legislative IDs.
        Matches patterns like ORD-2025-012, RES-2026-004, or PLAN-2026-99.
        """
        # Matches alphanumeric prefix (ORD|RES|PLAN) followed by year and number
        pattern = re.compile(r'\b(ORD|RES|PLAN)-\d{4}-\d+\b', re.IGNORECASE)
        matches = pattern.findall(text)
        
        # Re-construct full IDs because findall with group captures only the prefix
        # To get the full match:
        full_matches = []
        for match in re.finditer(pattern, text):
            full_matches.append(match.group(0).upper().strip())
            
        # De-duplicate while preserving order
        seen = set()
        return [x for x in full_matches if not (x in seen or seen.add(x))]

    def fetch_referenced_records(self, text: str) -> List[LegislativeRecord]:
        """
        Parses text for legislative ID references and fetches all matching records.
        """
        referenced_ids = self.extract_ids_from_text(text)
        records = []
        
        for record_id in referenced_ids:
            record = self.get_record(record_id)
            if record:
                records.append(record)
                
        return records
