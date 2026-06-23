"""
Unit Tests for the Citizen Feedback and Sentiment Analyzer Module
==================================================================
Verifies prompt configuration, mock public concern clustering fallbacks,
and structured OpenAI completions mapping citizen feedback.
"""

from unittest.mock import patch, MagicMock
import pytest
from summarization.citizen_feedback import CitizenFeedbackAnalyzer, PublicFeedbackReport, CitizenConcern


def test_build_analysis_prompt():
    """
    Asserts prompt builder instructs the LLM to cluster keywords and classify sentiment.
    """
    analyzer = CitizenFeedbackAnalyzer()
    dialogue = "[00:01:00] Citizen: There is a massive pothole on Elm St."

    prompt = analyzer.build_analysis_prompt(dialogue)

    assert dialogue in prompt
    assert "sentiment" in prompt.lower()
    assert "clustering" in prompt.lower()
    assert "impact" in prompt.lower()


def test_generate_mock_feedback_report():
    """
    Asserts mock fallback generates valid feedback clusters when keywords are present.
    """
    analyzer = CitizenFeedbackAnalyzer()

    dialogue = (
        "[00:00:01] Citizen A: We have no parking spaces around the new park.\n"
        "[00:01:00] Citizen B: Potholes on Elm Street are destroying our cars!"
    )

    report = analyzer._generate_mock_feedback_report(dialogue)

    assert report.meeting_title == "Greer City Council Public Hearing"
    assert report.overall_public_sentiment == "Divided"
    assert len(report.top_concerns) == 2

    categories = {c.concern_category: c for c in report.top_concerns}
    
    # Assert Zoning and Parking cluster
    assert "Zoning & Parking" in categories
    assert "parking" in categories["Zoning & Parking"].keywords
    assert categories["Zoning & Parking"].sentiment == "Concerned"
    assert categories["Zoning & Parking"].impact_rating == 4

    # Assert Roads and Infrastructure cluster
    assert "Roads & Infrastructure" in categories
    assert "potholes" in categories["Roads & Infrastructure"].keywords
    assert categories["Roads & Infrastructure"].sentiment == "Frustrated"
    assert categories["Roads & Infrastructure"].impact_rating == 5


def test_analyze_feedback_api():
    """
    Verifies that analyze_feedback executes structured completions
    using OpenAI client Beta parsing, returning the validated model.
    """
    analyzer = CitizenFeedbackAnalyzer()
    dialogue = "[00:01:00] SPEAKER: Public comment started."

    # Prepare Mock Report
    mock_concern = CitizenConcern(
        concern_category="Traffic safety",
        keywords=["speeding", "intersection"],
        sentiment="Concerned",
        summary="Requesting traffic lights",
        impact_rating=3
    )
    mock_report = PublicFeedbackReport(
        meeting_title="Greer Council Hearing",
        top_concerns=[mock_concern],
        overall_public_sentiment="Attentive"
    )

    # Mock chat completion parse response
    mock_choice = MagicMock()
    mock_choice.message.parsed = mock_report
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    with patch('summarization.citizen_feedback.OpenAI') as mock_openai_class:
        mock_client = mock_openai_class.return_value
        mock_client.beta.chat.completions.parse.return_value = mock_completion

        # Run
        res_report = analyzer.analyze_feedback(
            dialogue,
            openai_api_key="sk-mockkey"
        )

        # Assertions
        assert res_report is not None
        assert res_report.meeting_title == "Greer Council Hearing"
        assert len(res_report.top_concerns) == 1
        assert res_report.top_concerns[0].concern_category == "Traffic safety"
        assert res_report.top_concerns[0].impact_rating == 3
        
        # Verify call parameters
        mock_client.beta.chat.completions.parse.assert_called_once()
        call_kwargs = mock_client.beta.chat.completions.parse.call_args[1]
        assert call_kwargs["response_format"] == PublicFeedbackReport
        assert call_kwargs["model"] == "gpt-4o-mini"
