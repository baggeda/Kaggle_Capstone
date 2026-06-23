"""
Unit Tests for the Legislative Database Lookup Helper
=====================================================
Verifies SQLite schema creation, records upsert, exact and fuzzy retrieval,
and regular expression ID parsing on municipal meeting transcript inputs.
"""

from pathlib import Path
import pytest
from auditor.legislative_database import LegislativeRecord, LegislativeDatabaseHelper


@pytest.fixture
def temp_db(tmp_path):
    """
    Provides a temporary database helper instance pointing to a transient test file.
    """
    db_file = tmp_path / "test_legislative.db"
    helper = LegislativeDatabaseHelper(db_file)
    yield helper
    helper.close()


def test_db_upsert_and_retrieval(temp_db):
    """
    Asserts database writes and reads retrieve accurate structured data.
    """
    record = LegislativeRecord(
        legislative_id="ORD-2025-101",
        document_type="Ordinance",
        title="Park Zoning Amendment",
        summary="Amends zoning codes to expand park areas in Greer.",
        passed_date="2025-06-15",
        status="Passed"
    )

    # Test insert
    temp_db.upsert_record(record)
    retrieved = temp_db.get_record("ORD-2025-101")
    
    assert retrieved is not None
    assert retrieved.legislative_id == "ORD-2025-101"
    assert retrieved.document_type == "Ordinance"
    assert retrieved.title == "Park Zoning Amendment"
    assert retrieved.passed_date == "2025-06-15"

    # Test update (upsert)
    record.status = "Amended"
    record.title = "Park Zoning Amendment - Updated"
    temp_db.upsert_record(record)
    
    updated = temp_db.get_record("ORD-2025-101")
    assert updated is not None
    assert updated.status == "Amended"
    assert updated.title == "Park Zoning Amendment - Updated"

    # Non-existent ID lookup should yield None
    assert temp_db.get_record("ORD-9999-99") is None


def test_id_extraction_regex(temp_db):
    """
    Asserts regex parser extracts ordinance, resolution, and plan IDs,
    normalizes case, and de-duplicates correctly.
    """
    text = (
        "During debate, the clerk referenced Ordinance ord-2025-012 and resolution RES-2026-444.\n"
        "This was later modified under zoning plan Plan-2026-09. We also noted that ord-2025-012 "
        "is still key."
    )

    ids = temp_db.extract_ids_from_text(text)

    # Should capture 3 unique normalized IDs (ord-2025-012 is de-duplicated)
    assert len(ids) == 3
    assert "ORD-2025-012" in ids
    assert "RES-2026-444" in ids
    assert "PLAN-2026-09" in ids


def test_fetch_referenced_records(temp_db):
    """
    Asserts that parsing a dialogue stream successfully extracts IDs and
    returns all matching database records, while ignoring unknown references.
    """
    rec1 = LegislativeRecord(
        legislative_id="ORD-2025-001",
        document_type="Ordinance",
        title="Fiscal Budget 2025",
        summary="Defines Greer municipal budgets.",
        passed_date="2025-01-01",
        status="Passed"
    )
    rec2 = LegislativeRecord(
        legislative_id="RES-2026-002",
        document_type="Resolution",
        title="Greer Clean Water Action",
        summary="Directs Greer water department checks.",
        passed_date="2026-03-12",
        status="Passed"
    )

    # Seed DB
    temp_db.upsert_record(rec1)
    temp_db.upsert_record(rec2)

    # Dialogue referencing both seeded items, plus a non-existent item (ORD-9999-99)
    dialogue_stream = (
        "Under ORD-2025-001 we agreed to the budget allocations. Additionally, "
        "reconciliations will follow RES-2026-002. Note that ORD-9999-99 has not yet "
        "been drafted or introduced."
    )

    records = temp_db.fetch_referenced_records(dialogue_stream)

    # Should return exactly 2 matching records (ignoring the unknown ORD-9999-99)
    assert len(records) == 2
    ids = [r.legislative_id for r in records]
    assert "ORD-2025-001" in ids
    assert "RES-2026-002" in ids
    assert "ORD-9999-99" not in ids
