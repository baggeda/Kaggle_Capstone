"""
Unit Tests for the Post-Processing Audit Pipeline
=================================================
Verifies integrated auditing of compliance, timestamps, speaker presence, and bias checking.
"""

from unittest.mock import patch, MagicMock
import pytest
from transcription.audio_processor import DiarizedTranscript, AudioMetadata, DialogueLine
from summarization.agenda_aligner import AlignmentReport, AlignedSegment
from summarization.summary_engine import StructuredCivicSummary, CouncillorPerspective, AudienceResponse
from auditor.vote_parser import MeetingVotesReport, VoteOutcome, IndividualVote
from auditor.bias_checker import BiasEvaluation
from auditor.post_processor import PostProcessingPipeline


@pytest.fixture
def mock_transcript():
    metadata = AudioMetadata(file_path="meeting.wav", file_format="wav", file_size_bytes=100)
    lines = [
        DialogueLine(timestamp="[00:00:00]", speaker="CLERK", text="Roll call starts."),
        DialogueLine(timestamp="[00:00:10]", speaker="DAVIS", text="I am present."),
        DialogueLine(timestamp="[00:01:00]", speaker="MILLER", text="I oppose the park expansion."),
        DialogueLine(timestamp="[00:02:00]", speaker="CLERK", text="Meeting adjourned.")
    ]
    return DiarizedTranscript(metadata=metadata, lines=lines, full_text="[00:00:00] CLERK: Roll call. [00:00:10] DAVIS: Present.")


@pytest.fixture
def valid_alignment_report():
    return AlignmentReport(
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


@pytest.fixture
def valid_votes_report():
    return MeetingVotesReport(
        meeting_title="Greer Council Meeting",
        outcomes=[
            VoteOutcome(
                proposal_title="Park Expansion",
                result="Passed",
                ayes_count=1,
                nays_count=0,
                abstentions_count=0,
                voting_type="Voice Vote",
                votes_map=[
                    IndividualVote(councillor_name="Davis", vote="Aye")
                ]
            )
        ]
    )


@pytest.fixture
def valid_summary():
    return StructuredCivicSummary(
        agenda_item_title="Greer Park Expansion",
        debate_points=[],
        councillor_perspectives=[
            CouncillorPerspective(
                councillor_name="Councillor Davis",
                stance="Support",
                vocal_opposition=False
            )
        ],
        audience_response=AudienceResponse(general_sentiment="Mixed"),
        summary_narrative="Councillor Davis supported the park expansion."
    )


def test_post_processing_pipeline_success(
    mock_transcript,
    valid_alignment_report,
    valid_votes_report,
    valid_summary
):
    """
    Verifies that pipeline passes when all checks are valid and bias evaluation is neutral.
    """
    pipeline = PostProcessingPipeline()

    # Create a neutral bias evaluation mock response
    mock_eval = BiasEvaluation(
        is_neutral=True,
        loaded_language_instances=[],
        partisan_tone_issues=[],
        misattributions=[],
        revised_summary="Councillor Davis supported the park expansion.",
        auditor_feedback="Neutrality check passed."
    )

    with patch('auditor.bias_checker.OpenAI') as mock_openai_class:
        mock_client = mock_openai_class.return_value
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(parsed=mock_eval))]
        mock_client.beta.chat.completions.parse.return_value = mock_completion

        report = pipeline.process(
            summary=valid_summary,
            alignment_report=valid_alignment_report,
            votes_report=valid_votes_report,
            raw_transcript=mock_transcript,
            openai_api_key="sk-mockkey"
        )

        assert report.is_valid is True
        assert report.alignment_validation.is_valid is True
        assert report.summary_validation.is_valid is True
        assert report.votes_validation.is_valid is True
        assert report.bias_evaluation.is_neutral is True
        assert report.final_revised_summary == "Councillor Davis supported the park expansion."


def test_post_processing_pipeline_alignment_failure(
    mock_transcript,
    valid_votes_report,
    valid_summary
):
    """
    Verifies that pipeline fails when alignment report contains invalid bounds.
    """
    pipeline = PostProcessingPipeline()

    invalid_alignment_report = AlignmentReport(
        meeting_title="Greer Council",
        segments=[
            AlignedSegment(
                agenda_item_id="1",
                agenda_title="Roll Call",
                start_timestamp="[00:02:10]",  # Out of bounds
                end_timestamp="[00:02:30]"
            )
        ]
    )

    # Mock bias evaluation
    mock_eval = BiasEvaluation(
        is_neutral=True,
        loaded_language_instances=[],
        partisan_tone_issues=[],
        misattributions=[],
        revised_summary="Councillor Davis supported the park expansion.",
        auditor_feedback="Neutrality check passed."
    )

    with patch('auditor.bias_checker.OpenAI') as mock_openai_class:
        mock_client = mock_openai_class.return_value
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(parsed=mock_eval))]
        mock_client.beta.chat.completions.parse.return_value = mock_completion

        report = pipeline.process(
            summary=valid_summary,
            alignment_report=invalid_alignment_report,
            votes_report=valid_votes_report,
            raw_transcript=mock_transcript,
            openai_api_key="sk-mockkey"
        )

        assert report.is_valid is False
        assert report.alignment_validation.is_valid is False
        assert report.summary_validation.is_valid is True


def test_post_processing_pipeline_bias_failure(
    mock_transcript,
    valid_alignment_report,
    valid_votes_report,
    valid_summary
):
    """
    Verifies that pipeline fails when bias check detects loaded language and suggests revision.
    """
    pipeline = PostProcessingPipeline()

    # Mock biased evaluation
    mock_eval = BiasEvaluation(
        is_neutral=False,
        loaded_language_instances=["corrupt"],
        partisan_tone_issues=["Slanted description"],
        misattributions=[],
        revised_summary="Councillor Davis supported the park expansion.",
        auditor_feedback="Audit failed due to loaded language."
    )

    with patch('auditor.bias_checker.OpenAI') as mock_openai_class:
        mock_client = mock_openai_class.return_value
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(parsed=mock_eval))]
        mock_client.beta.chat.completions.parse.return_value = mock_completion

        report = pipeline.process(
            summary=valid_summary,
            alignment_report=valid_alignment_report,
            votes_report=valid_votes_report,
            raw_transcript=mock_transcript,
            openai_api_key="sk-mockkey"
        )

        assert report.is_valid is False
        assert report.alignment_validation.is_valid is True
        assert report.bias_evaluation.is_neutral is False
        assert report.final_revised_summary == "Councillor Davis supported the park expansion."


def test_post_processing_pipeline_auto_corrects_alignment_offline(
    mock_transcript,
    valid_votes_report,
    valid_summary
):
    """
    Verifies that pipeline automatically corrects alignment timestamps offline and passes on the next attempt.
    """
    pipeline = PostProcessingPipeline()

    invalid_alignment_report = AlignmentReport(
        meeting_title="Greer Council",
        segments=[
            AlignedSegment(
                agenda_item_id="1",
                agenda_title="Roll Call",
                start_timestamp="[00:02:10]",  # Out of bounds
                end_timestamp="[00:02:30]"
            )
        ]
    )

    # Use None API key to force offline mock fallback correction
    report = pipeline.process(
        summary=valid_summary,
        alignment_report=invalid_alignment_report,
        votes_report=valid_votes_report,
        raw_transcript=mock_transcript,
        openai_api_key=None
    )

    assert report.is_valid is True  # Should pass after auto-correction
    assert report.alignment_validation.is_valid is True


def test_post_processing_pipeline_auto_corrects_bias_offline(
    mock_transcript,
    valid_alignment_report,
    valid_votes_report
):
    """
    Verifies that pipeline automatically corrects biased summaries offline and passes on the next attempt.
    """
    pipeline = PostProcessingPipeline()

    # Create a biased summary narrative
    biased_summary = StructuredCivicSummary(
        agenda_item_title="Greer Park Expansion",
        debate_points=[],
        councillor_perspectives=[
            CouncillorPerspective(
                councillor_name="Councillor Davis",
                stance="Support",
                vocal_opposition=False
            )
        ],
        audience_response=AudienceResponse(general_sentiment="Mixed"),
        summary_narrative="The council approved the corrupt zoning contract."  # Contains "corrupt" -> biased
    )

    # Use None API key to force offline mock bias evaluation & correction
    report = pipeline.process(
        summary=biased_summary,
        alignment_report=valid_alignment_report,
        votes_report=valid_votes_report,
        raw_transcript=mock_transcript,
        openai_api_key=None
    )

    assert report.is_valid is True  # Should pass on the second attempt
    assert report.bias_evaluation.is_neutral is True
    assert "corrupt" not in report.final_revised_summary

