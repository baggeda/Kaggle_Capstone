"""
Legislative Database Seeding Script
===================================
Pre-populates the municipal legislative records database with historical
ordinances, resolutions, and zoning plans for Greer, Greenville, and Spartanburg.
"""

import sys
from pathlib import Path

# Add project root to path to enable importing from auditor
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from auditor.legislative_database import LegislativeRecord, LegislativeDatabaseHelper


def seed_database(db_path: Path) -> None:
    """
    Seeds the target SQLite database with realistic historical municipal records.
    """
    print(f"Initializing database at: {db_path}")
    helper = LegislativeDatabaseHelper(db_path)

    historical_records = [
        # Greer Records
        LegislativeRecord(
            legislative_id="ORD-2025-012",
            document_type="Ordinance",
            title="Greer Park Expansion and Zoning Amendment",
            summary="Amends the municipal zoning map to authorize expansion of public parks and greenways, allocating $150,000 for parking infrastructure.",
            passed_date="2025-06-15",
            status="Passed"
        ),
        LegislativeRecord(
            legislative_id="ORD-2025-001",
            document_type="Ordinance",
            title="Greer Fiscal Year 2025 Annual Operating Budget",
            summary="Adopts the annual operating budget for the City of Greer, SC, for the fiscal year beginning January 1, 2025.",
            passed_date="2024-12-10",
            status="Passed"
        ),
        LegislativeRecord(
            legislative_id="ORD-2025-045",
            document_type="Ordinance",
            title="Greer Noise Control and Abatement Ordinance",
            summary="Regulates decibel levels for commercial and residential zones, setting curfew hours and penalties for violations.",
            passed_date="2025-09-08",
            status="Passed"
        ),
        
        # Greenville Records
        LegislativeRecord(
            legislative_id="RES-2026-002",
            document_type="Resolution",
            title="Greenville Water Quality Monitoring Initiative",
            summary="Directs the environmental services division to perform bi-monthly inspections of municipal water pipelines and report findings.",
            passed_date="2026-03-12",
            status="Passed"
        ),
        LegislativeRecord(
            legislative_id="RES-2026-015",
            document_type="Resolution",
            title="Greenville Traffic Flow and Pedestrian Safety Resolution",
            summary="Authorizes installation of traffic control signals and high-visibility crosswalks along Main Street.",
            passed_date="2026-05-14",
            status="Passed"
        ),

        # Spartanburg Records
        LegislativeRecord(
            legislative_id="PLAN-2026-009",
            document_type="Zoning Plan",
            title="Spartanburg Downtown Commercial Revitalization Plan",
            summary="Approves the master zoning development plan for the central downtown business district to encourage high-density mixed-use retail.",
            passed_date="2026-02-18",
            status="Passed"
        ),
        LegislativeRecord(
            legislative_id="RES-2025-088",
            document_type="Resolution",
            title="Establishment of the Spartanburg Affordable Housing Trust Fund",
            summary="Creates a dedicated trust fund to subsidize low-to-moderate-income multi-family housing projects.",
            passed_date="2025-11-20",
            status="Passed"
        )
    ]

    print(f"Seeding {len(historical_records)} legislative records...")
    for record in historical_records:
        helper.upsert_record(record)
        print(f" - Seeded {record.legislative_id}: {record.title}")

    helper.close()
    print("Database seeding completed successfully.")


if __name__ == "__main__":
    # Use default path or override via argument
    default_db = PROJECT_ROOT / "data" / "legislative_records.db"
    
    target_path = default_db
    if len(sys.argv) > 1:
        target_path = Path(sys.argv[1])
        
    seed_database(target_path)
