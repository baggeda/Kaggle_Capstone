"""
Greenville City Council Meeting Pipeline Run
===========================================
Executes the end-to-end civic engagement pipeline for a mock Greenville City Council meeting,
running media routing, diarized transcription, agenda alignment, voting extraction,
legislative database lookup, and post-processing validation/auto-correction.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJECT_ROOT = Path("e:/Users/Dan/Documents/Antigravity/Kaggle_Capstone")
sys.path.insert(0, str(PROJECT_ROOT))

from transcription.audio_processor import AudioMetadata, DiarizedTranscript, DialogueLine, DiarizationSegment, TranscriptionSegment
from transcription.pipeline import DiarizedTranscriptionPipeline
from summarization.agenda_aligner import AgendaAligner, AlignmentReport, AlignedSegment
from summarization.summary_engine import SummaryEngine, StructuredCivicSummary, CouncillorPerspective, AudienceResponse, DebatePoint
from auditor.vote_parser import VoteParser, MeetingVotesReport, VoteOutcome, IndividualVote
from auditor.legislative_database import LegislativeDatabaseHelper
from auditor.post_processor import PostProcessingPipeline, PostProcessingReport
from auditor.bias_checker import BiasEvaluation


def main():
    print("--- STARTING GREENVILLE CITY COUNCIL PIPELINE ---")
    
    # 1. Create a dummy meeting audio file in data/raw
    raw_dir = PROJECT_ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    audio_path = raw_dir / "greenville_council_2026_06_23.wav"
    audio_path.write_bytes(b"RIFFdummywavbytes" * 50)
    print(f"Created temporary mock audio file at: {audio_path}")

    # 2. Mock Transcription and Diarization responses
    # Dialogue involves Knox White, Lillian Flemming, and Wil Brasington
    mock_diarizer_segs = [
        DiarizationSegment(start=0.0, end=15.0, speaker="KNOX WHITE"),
        DiarizationSegment(start=15.0, end=45.0, speaker="LILLIAN FLEMMING"),
        DiarizationSegment(start=45.5, end=75.0, speaker="WIL BRASINGTON"),
        DiarizationSegment(start=75.0, end=90.0, speaker="KNOX WHITE")
    ]

    mock_whisper_segs = [
        TranscriptionSegment(start=1.0, end=14.0, text="Welcome to the Greenville City Council meeting. We are in session."),
        TranscriptionSegment(start=16.0, end=43.0, text="Thank you, Mayor. Under RES-2026-002, we are expanding our water quality monitoring pipelines."),
        TranscriptionSegment(start=46.0, end=74.0, text="I support that, but under RES-2026-015 we must also adopt the Main Street traffic safety project."),
        TranscriptionSegment(start=76.0, end=88.0, text="Great. The motion for both resolutions is adopted unanimously. Meeting adjourned.")
    ]

    # 3. Setup the Agenda text
    agenda_text = (
        "Greenville City Council Meeting Agenda - June 23, 2026\n"
        "Item 1. Greenville Water Quality Monitoring Initiative (RES-2026-002)\n"
        "Item 2. Traffic Flow and Pedestrian Safety Resolution (RES-2026-015)"
    )

    # 4. Instantiate pipeline modules
    pipeline = DiarizedTranscriptionPipeline()
    aligner = AgendaAligner()
    summary_engine = SummaryEngine()
    vote_parser = VoteParser()
    db_helper = LegislativeDatabaseHelper(PROJECT_ROOT / "data" / "legislative_records.db")
    post_processor = PostProcessingPipeline()

    print("\n--- STEP 1: RUN DIARIZED TRANSCRIPTION PIPELINE ---")
    with patch.object(pipeline.diarizer, 'diarize', return_value=mock_diarizer_segs), \
         patch.object(pipeline, '_get_mock_whisper_segments', return_value=mock_whisper_segs):
        
        transcript = pipeline.run(audio_path, openai_api_key=None)
        print("Transcript generated successfully.")
        print(f"Total transcript lines: {len(transcript.lines)}")
        print("Transcript Preview:")
        print(transcript.full_text)

    print("\n--- STEP 2: RUN AGENDA ALIGNMENT ---")
    # Generate alignment report
    alignment_report = aligner.align_dialogue_to_agenda(
        dialogue_text=transcript.full_text,
        agenda_text=agenda_text,
        openai_api_key=None
    )
    # Customize mock alignment with real Greenville titles
    alignment_report.meeting_title = "Greenville City Council Meeting"
    alignment_report.segments = [
        AlignedSegment(
            agenda_item_id="Item 1",
            agenda_title="Greenville Water Quality Monitoring Initiative",
            start_timestamp="[00:00:15]",
            end_timestamp="[00:00:45]",
            discussion_points=["Lillian Flemming introduced the water quality project."],
            decisions_made=["Approved the initiative under RES-2026-002."]
        ),
        AlignedSegment(
            agenda_item_id="Item 2",
            agenda_title="Traffic Flow and Pedestrian Safety Resolution",
            start_timestamp="[00:00:45]",
            end_timestamp="[00:01:14]",
            discussion_points=["Wil Brasington introduced the traffic safety project."],
            decisions_made=["Adopted the Main Street traffic signals under RES-2026-015."]
        )
    ]
    print(f"Aligned {len(alignment_report.segments)} agenda segments.")

    print("\n--- STEP 3: RUN STRUCTURED SUMMARIZATION ---")
    structured_summary = summary_engine.generate_structured_summary(
        dialogue_text=transcript.full_text,
        agenda_item_title="Water Quality & Traffic Safety Initiatives",
        openai_api_key=None
    )
    # Inject real Greenville councillor names to check speaker attributions
    structured_summary.councillor_perspectives = [
        CouncillorPerspective(
            councillor_name="Lillian Flemming",
            stance="Support",
            key_arguments=["Expanding monitor testing ensures clean water."],
            vocal_opposition=False
        ),
        CouncillorPerspective(
            councillor_name="Wil Brasington",
            stance="Support",
            key_arguments=["Main Street safety signals are overdue."],
            vocal_opposition=False
        )
    ]
    # Let's inject a loaded word to trigger the bias auditor's auto-correction loop
    structured_summary.summary_narrative = (
        "Lillian Flemming proposed the water monitoring plan, while Wil Brasington supported "
        "the traffic safety resolution which was a foolish project."
    )
    print("Structured Civic Summary generated.")

    print("\n--- STEP 4: RUN VOTE PARSING ---")
    votes_report = vote_parser.parse_votes_from_dialogue(
        dialogue_text=transcript.full_text,
        openai_api_key=None
    )
    # Populate the votes report with Greenville councillors
    votes_report.meeting_title = "Greenville City Council Meeting"
    votes_report.outcomes = [
        VoteOutcome(
            proposal_id="RES-2026-002",
            proposal_title="Water Quality Monitoring Initiative",
            result="Passed",
            ayes_count=7,
            nays_count=0,
            abstentions_count=0,
            voting_type="Voice Vote",
            votes_map=[
                IndividualVote(councillor_name="Knox White", vote="Aye"),
                IndividualVote(councillor_name="Lillian Flemming", vote="Aye"),
                IndividualVote(councillor_name="Wil Brasington", vote="Aye"),
                IndividualVote(councillor_name="Dorothy Dowe", vote="Aye"),
                IndividualVote(councillor_name="Ken Gibson", vote="Aye"),
                IndividualVote(councillor_name="Tina Belge", vote="Aye"),
                IndividualVote(councillor_name="John DeWorken", vote="Aye")
            ]
        )
    ]
    print(f"Extracted vote results for {len(votes_report.outcomes)} motions.")

    print("\n--- STEP 5: LEGISLATIVE DATABASE LOOKUP ---")
    # Parse legislative references in the dialogue
    referenced_ids = db_helper.extract_ids_from_text(transcript.full_text)
    print(f"Referenced Legislative IDs in dialogue: {referenced_ids}")
    
    historical_records = db_helper.fetch_referenced_records(transcript.full_text)
    print(f"Found {len(historical_records)} historical records in SQLite database:")
    for record in historical_records:
        print(f" - {record.legislative_id} ({record.document_type}): {record.title}")
        print(f"   Summary: {record.summary}")

    print("\n--- STEP 6: RUN COMPLIANCE POST-PROCESSING AUDIT & AUTO-CORRECTION LOOP ---")
    # Run the processor. Since the summary_narrative contains the loaded word "foolish", 
    # the bias checker will flag it, and the pipeline will automatically trigger 
    # auto-correction to remove the bias.
    report = post_processor.process(
        summary=structured_summary,
        alignment_report=alignment_report,
        votes_report=votes_report,
        raw_transcript=transcript,
        openai_api_key=None
    )

    print("\n--- AUDIT RESULTS ---")
    print(f"Audit Is Valid: {report.is_valid}")
    print(f"Alignment Valid: {report.alignment_validation.is_valid}")
    print(f"Summary Valid: {report.summary_validation.is_valid}")
    print(f"Votes Valid: {report.votes_validation.is_valid}")
    print(f"Bias Neutral: {report.bias_evaluation.is_neutral}")
    print("\nFinal Clean Summary Narrative:")
    print(report.final_revised_summary)

    # Clean up mock file
    if audio_path.exists():
        audio_path.unlink()


if __name__ == "__main__":
    main()
