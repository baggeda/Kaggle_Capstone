"""
Unit Tests for the LLM Dialogue-to-Agenda Alignment Module
===========================================================
Verifies prompt building, mock alignment fallback matching, and
structured OpenAI API completions for dialogue segment alignment.
"""

from unittest.mock import patch, MagicMock
import pytest
from summarization.agenda_aligner import AgendaAligner, AlignmentReport, AlignedSegment


def test_build_prompt():
    """
    Asserts prompt builder aggregates dialogue and agenda text.
    """
    aligner = AgendaAligner()
    dialogue = "[00:01:00] SPEAKER_01: The budget amendment has passed."
    agenda = "Item 2: Budget Reconciliation vote."

    prompt = aligner.build_prompt(dialogue, agenda)
    
    assert dialogue in prompt
    assert agenda in prompt
    assert "segment" in prompt.lower()
    assert "align" in prompt.lower()


def test_generate_mock_alignment():
    """
    Asserts mock fallback generates correct segments when keywords are present.
    """
    aligner = AgendaAligner()

    agenda = "Greer City Council Agenda\nItem 1: Greer Park Expansion\nItem 2: Budget Reconciliation"
    dialogue = (
        "[00:00:01] SPEAKER_01: Welcome to the park expansion hearing.\n"
        "[00:01:00] SPEAKER_02: Let us vote on the budget reconciliation bill."
    )

    report = aligner._generate_mock_alignment(dialogue, agenda)

    assert report.meeting_title == "Greer City Council Agenda"
    assert len(report.segments) == 2
    
    # Assert first segment (Park)
    assert report.segments[0].agenda_item_id == "Item 1"
    assert "Park Expansion" in report.segments[0].agenda_title
    assert report.segments[0].start_timestamp == "[00:00:01]"
    assert "Planning Commission" in report.segments[0].decisions_made[0]

    # Assert second segment (Budget)
    assert report.segments[1].agenda_item_id == "Item 2"
    assert "Budget" in report.segments[1].agenda_title
    assert report.segments[1].start_timestamp == "[00:01:00]"
    assert "Adopted" in report.segments[1].decisions_made[0]


def test_align_dialogue_to_agenda_api(tmp_path):
    """
    Verifies that align_dialogue_to_agenda executes structured completions
    using OpenAI client Beta parsing, returning the validated model.
    """
    aligner = AgendaAligner()

    agenda = "Spartanburg City Council"
    dialogue = "[00:01:00] SPEAKER_01: Welcome."

    # Prepare Mock Report
    mock_aligned_segment = AlignedSegment(
        agenda_item_id="Item 3",
        agenda_title="Zoning amendment",
        start_timestamp="[00:01:00]",
        end_timestamp="[00:01:30]",
        discussion_points=["Zoning maps update discussion"],
        decisions_made=["Adopted zoning change"]
    )
    mock_report = AlignmentReport(
        meeting_title="Spartanburg Council",
        segments=[mock_aligned_segment]
    )

    # Mock chat completion parse response
    mock_choice = MagicMock()
    mock_choice.message.parsed = mock_report
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    with patch('summarization.agenda_aligner.OpenAI') as mock_openai_class:
        mock_client = mock_openai_class.return_value
        mock_client.beta.chat.completions.parse.return_value = mock_completion

        # Run
        res_report = aligner.align_dialogue_to_agenda(
            dialogue,
            agenda,
            openai_api_key="sk-mockkey"
        )

        # Assertions
        assert res_report is not None
        assert res_report.meeting_title == "Spartanburg Council"
        assert len(res_report.segments) == 1
        assert res_report.segments[0].agenda_item_id == "Item 3"
        assert res_report.segments[0].decisions_made[0] == "Adopted zoning change"
        
        # Verify call parameters
        mock_client.beta.chat.completions.parse.assert_called_once()
        call_kwargs = mock_client.beta.chat.completions.parse.call_args[1]
        assert call_kwargs["response_format"] == AlignmentReport
        assert call_kwargs["model"] == "gpt-4o-mini"
