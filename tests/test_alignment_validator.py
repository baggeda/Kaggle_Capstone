"""
Unit Tests for the Summary Alignment Validator
===============================================
Verifies timestamp parsing, bounds validation, chronological checking, and
speaker attribution auditing across summaries, alignments, and voting records.
"""

import pytest
from pydantic import BaseModel
from transcription.audio_processor import DiarizedTranscript, AudioMetadata, DialogueLine
from summarization.agenda_aligner import AlignmentReport, AlignedSegment
from summarization.summary_engine import StructuredCivicSummary, CouncillorPerspective, AudienceResponse
from auditor.vote_parser import MeetingVotesReport, VoteOutcome, IndividualVote
from auditor.alignment_validator import SummaryAlignmentValidator


@pytest.fixture
def validator():
    return SummaryAlignmentValidator()


@pytest.fixture
def mock_transcript():
    """
    Returns a mock transcript with a total duration of 2 minutes (120 seconds).
    Speakers: DAVIS, MILLER, CLERK.
    """
    metadata = AudioMetadata(file_path="meeting.wav", file_format="wav", file_size_bytes=100)
    lines = [
        DialogueLine(timestamp="[00:00:00]", speaker="CLERK", text="Roll call starts."),
        DialogueLine(timestamp="[00:00:10]", speaker="DAVIS", text="I am present."),
        DialogueLine(timestamp="[00:01:00]", speaker="MILLER", text="I oppose the park expansion."),
        DialogueLine(timestamp="[00:02:00]", speaker="CLERK", text="Meeting adjourned.")
    ]
    return DiarizedTranscript(metadata=metadata, lines=lines, full_text="")


def test_timestamp_parsing(validator):
    """
    Asserts timestamp parsing resolves seconds correctly.
    """
    assert validator.parse_timestamp_to_seconds("00:00:00") == 0.0
    assert validator.parse_timestamp_to_seconds("[00:01:30]") == 90.0
    assert validator.parse_timestamp_to_seconds("[02:00:00]") == 7200.0
    assert validator.parse_timestamp_to_seconds("invalid") is None


def test_validate_alignment_report(validator, mock_transcript):
    """
    Verifies validation checks for chronological ordering and temporal bounds.
    """
    # 1. Valid report
    valid_report = AlignmentReport(
        meeting_title="Greer Council",
        segments=[
            AlignedSegment(
                agenda_item_id="1",
                agenda_title="Roll Call",
                start_timestamp="[00:00:00]",
                end_timestamp="[00:00:30]"
            )
        ]
    )
    res_valid = validator.validate_alignment_report(valid_report, mock_transcript)
    assert res_valid.is_valid is True
    assert len(res_valid.issues) == 0

    # 2. Out of order segment (start > end)
    invalid_order_report = AlignmentReport(
        meeting_title="Greer Council",
        segments=[
            AlignedSegment(
                agenda_item_id="2",
                agenda_title="Budget debate",
                start_timestamp="[00:01:30]",
                end_timestamp="[00:01:00]"
            )
        ]
    )
    res_order = validator.validate_alignment_report(invalid_order_report, mock_transcript)
    assert res_order.is_valid is False
    assert len(res_order.issues) == 1
    assert res_order.issues[0].issue_type == "out_of_order"

    # 3. Out of bounds segment (exceeds transcript duration of 120s)
    invalid_bounds_report = AlignmentReport(
        meeting_title="Greer Council",
        segments=[
            AlignedSegment(
                agenda_item_id="3",
                agenda_title="Adjournment",
                start_timestamp="[00:02:10]",
                end_timestamp="[00:02:30]"
            )
        ]
    )
    res_bounds = validator.validate_alignment_report(invalid_bounds_report, mock_transcript)
    assert res_bounds.is_valid is False
    assert len(res_bounds.issues) == 1
    assert res_bounds.issues[0].issue_type == "out_of_bounds"


def test_validate_structured_summary_speakers(validator, mock_transcript):
    """
    Verifies validation checks flag councillors who did not speak in the transcript.
    """
    # Davis and Miller are in the transcript. Smith is not.
    summary = StructuredCivicSummary(
        agenda_item_title="Greer Park Expansion",
        debate_points=[],
        councillor_perspectives=[
            CouncillorPerspective(
                councillor_name="Councillor Davis",
                stance="Support",
                vocal_opposition=False
            ),
            CouncillorPerspective(
                councillor_name="Councillor Smith",
                stance="Oppose",
                vocal_opposition=True
            )
        ],
        audience_response=AudienceResponse(general_sentiment="Mixed"),
        summary_narrative="Davis and Smith debated."
    )

    res = validator.validate_structured_summary(summary, mock_transcript)

    assert res.is_valid is False
    assert len(res.issues) == 1
    assert res.issues[0].issue_type == "unknown_speaker"
    assert res.issues[0].entity_id == "Councillor Smith"


def test_validate_votes_report_speakers(validator, mock_transcript):
    """
    Verifies validation checks flag voting councillors not present or speaking,
    while ignoring legitimate absentees.
    """
    # Davis and Miller are in the transcript. Brown is not in the transcript, but is marked 'Absent'.
    # Jones is not in the transcript, but is marked as voting 'Aye' (invalid).
    votes_report = MeetingVotesReport(
        meeting_title="Greer Council Meeting",
        outcomes=[
            VoteOutcome(
                proposal_title="Park Expansion",
                result="Passed",
                ayes_count=2,
                nays_count=0,
                abstentions_count=0,
                voting_type="Voice Vote",
                votes_map=[
                    IndividualVote(councillor_name="Davis", vote="Aye"),
                    IndividualVote(councillor_name="Brown", vote="Absent"),
                    IndividualVote(councillor_name="Jones", vote="Aye")
                ]
            )
        ]
    )

    res = validator.validate_votes_report(votes_report, mock_transcript)

    assert res.is_valid is False
    assert len(res.issues) == 1
    assert res.issues[0].issue_type == "unknown_speaker"
    assert res.issues[0].entity_id == "Jones"
