"""
Unit Tests for the Structured Civic Summary Prompt Engine
==========================================================
Verifies prompt configuration, mock summary fallbacks, and
structured OpenAI API completions for council debate summaries.
"""

from unittest.mock import patch, MagicMock
import pytest
from summarization.summary_engine import SummaryEngine, StructuredCivicSummary, CouncillorPerspective


def test_build_structured_prompt():
    """
    Asserts prompt builder instructs the LLM to capture opposition and sentiment.
    """
    engine = SummaryEngine()
    dialogue = "[00:01:00] Davis: I support the park. [00:01:10] Miller: Too expensive!"
    title = "Greer Park Expansion"

    prompt = engine.build_structured_prompt(dialogue, title)

    assert title in prompt
    assert dialogue in prompt
    assert "councillor" in prompt.lower()
    assert "opposition" in prompt.lower()
    assert "audience" in prompt.lower()
    assert "sentiment" in prompt.lower()


def test_generate_mock_structured_summary():
    """
    Asserts mock fallback generates valid structured summary models.
    """
    engine = SummaryEngine()
    summary = engine._generate_mock_structured_summary("Greer Park Expansion")

    assert summary.agenda_item_title == "Greer Park Expansion"
    assert len(summary.debate_points) == 1
    
    # Assert councillor perspectives (Miller is vocal opposition, Davis is supportive)
    councillors = {c.councillor_name: c for c in summary.councillor_perspectives}
    assert "Councillor Davis" in councillors
    assert councillors["Councillor Davis"].vocal_opposition is False

    assert "Councillor Miller" in councillors
    assert councillors["Councillor Miller"].vocal_opposition is True
    assert councillors["Councillor Miller"].stance == "Oppose"

    # Assert audience response details
    assert summary.audience_response.general_sentiment == "Mixed"
    assert "Noise during construction." in summary.audience_response.public_speaker_concerns


def test_generate_structured_summary_api():
    """
    Verifies that generate_structured_summary executes structured completions
    using OpenAI client Beta parsing, returning the validated model.
    """
    engine = SummaryEngine()
    dialogue = "[00:01:00] SPEAKER: Hello."
    title = "Zoning Bill"

    # Prepare Mock Report
    mock_perspective = CouncillorPerspective(
        councillor_name="Councillor Davis",
        stance="Support",
        key_arguments=["Good zoning plan."],
        vocal_opposition=False
    )
    mock_summary = engine._generate_mock_structured_summary("Zoning Bill")
    mock_summary.councillor_perspectives = [mock_perspective]

    # Mock chat completion parse response
    mock_choice = MagicMock()
    mock_choice.message.parsed = mock_summary
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    with patch('summarization.summary_engine.OpenAI') as mock_openai_class:
        mock_client = mock_openai_class.return_value
        mock_client.beta.chat.completions.parse.return_value = mock_completion

        # Run
        res_summary = engine.generate_structured_summary(
            dialogue,
            title,
            openai_api_key="sk-mockkey"
        )

        # Assertions
        assert res_summary is not None
        assert res_summary.agenda_item_title == "Zoning Bill"
        assert len(res_summary.councillor_perspectives) == 1
        assert res_summary.councillor_perspectives[0].councillor_name == "Councillor Davis"
        assert res_summary.councillor_perspectives[0].vocal_opposition is False
        
        # Verify call parameters
        mock_client.beta.chat.completions.parse.assert_called_once()
        call_kwargs = mock_client.beta.chat.completions.parse.call_args[1]
        assert call_kwargs["response_format"] == StructuredCivicSummary
        assert call_kwargs["model"] == "gpt-4o-mini"
