"""
Unit Tests for the Bias & Neutrality Auditing Check-Agent
========================================================
Verifies prompt configuration, mock bias/misattribution detection fallbacks,
and structured OpenAI completions mapping ombudsman audits.
"""

from unittest.mock import patch, MagicMock
import pytest
from auditor.bias_checker import SummaryBiasChecker, BiasEvaluation


def test_build_audit_prompt():
    """
    Asserts prompt builder configures ombudsman rules and original dialogue block.
    """
    checker = SummaryBiasChecker()
    summary = "The council made a foolish decision."
    dialogue = "Davis: Let's defer."

    prompt = checker.build_audit_prompt(summary, dialogue)

    assert summary in prompt
    assert dialogue in prompt
    assert "loaded language" in prompt.lower()
    assert "misattribution" in prompt.lower()
    assert "ombudsman" in prompt.lower()


def test_generate_mock_bias_evaluation():
    """
    Asserts mock fallback audits loaded language and attribution errors.
    """
    checker = SummaryBiasChecker()

    # 1. Test loaded language detection
    summary_biased = "The mayor approved the corrupt zoning contract."
    eval_biased = checker._generate_mock_bias_evaluation(summary_biased)
    
    assert eval_biased.is_neutral is False
    assert "corrupt" in eval_biased.loaded_language_instances
    assert "[AUDITED AND REVISED SUMMARY]" in eval_biased.revised_summary

    # 2. Test misattribution detection
    summary_misattributed = "Davis opposed the bill."
    dialogue = "Davis: I support the funding allocations."
    
    eval_misattrib = checker._generate_mock_bias_evaluation(summary_misattributed, dialogue)
    assert eval_misattrib.is_neutral is False
    assert len(eval_misattrib.misattributions) == 1
    assert "Davis" in eval_misattrib.misattributions[0]

    # 3. Test neutral summary (passes audit)
    summary_neutral = "The council voted to adopt the zoning code amendment."
    eval_neutral = checker._generate_mock_bias_evaluation(summary_neutral)
    assert eval_neutral.is_neutral is True
    assert len(eval_neutral.loaded_language_instances) == 0


def test_audit_summary_api():
    """
    Verifies that audit_summary executes structured completions
    using OpenAI client Beta parsing, returning the validated model.
    """
    checker = SummaryBiasChecker()
    summary = "Neutral summary text."

    # Prepare Mock Report
    mock_eval = BiasEvaluation(
        is_neutral=True,
        loaded_language_instances=[],
        partisan_tone_issues=[],
        misattributions=[],
        revised_summary="Neutral summary text.",
        auditor_feedback="Neutrality check passed."
    )

    # Mock chat completion parse response
    mock_choice = MagicMock()
    mock_choice.message.parsed = mock_eval
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    with patch('auditor.bias_checker.OpenAI') as mock_openai_class:
        mock_client = mock_openai_class.return_value
        mock_client.beta.chat.completions.parse.return_value = mock_completion

        # Run
        res_eval = checker.audit_summary(
            summary,
            openai_api_key="sk-mockkey"
        )

        # Assertions
        assert res_eval is not None
        assert res_eval.is_neutral is True
        assert res_eval.revised_summary == "Neutral summary text."
        assert res_eval.auditor_feedback == "Neutrality check passed."
        
        # Verify call parameters
        mock_client.beta.chat.completions.parse.assert_called_once()
        call_kwargs = mock_client.beta.chat.completions.parse.call_args[1]
        assert call_kwargs["response_format"] == BiasEvaluation
        assert call_kwargs["model"] == "gpt-4o-mini"
