"""
Unit Tests for the LLM Vote Outcome Parser Module
=================================================
Verifies prompt configuration, mock voting outcome fallbacks, and
structured OpenAI API completions for mapping votes to council members.
"""

from unittest.mock import patch, MagicMock
import pytest
from auditor.vote_parser import VoteParser, MeetingVotesReport, VoteOutcome, IndividualVote


def test_build_vote_prompt():
    """
    Asserts prompt builder instructs the LLM to map votes to councillors.
    """
    parser = VoteParser()
    dialogue = "[00:01:00] Miller: I vote nay. [00:01:10] Davis: Aye."
    
    prompt = parser.build_vote_prompt(dialogue)

    assert dialogue in prompt
    assert "councillor" in prompt.lower()
    assert "ayes" in prompt.lower()
    assert "nays" in prompt.lower()
    assert "roll call" in prompt.lower()


def test_generate_mock_votes_report():
    """
    Asserts mock fallback generates valid voting reports with mapped votes.
    """
    parser = VoteParser()

    dialogue = (
        "[00:00:01] Clerk: Davis? Davis: Aye. Clerk: Miller? Miller: Nay.\n"
        "[00:01:00] Mayor: Let us vote on the budget reconciliation bill. [Chorus of Ayes]"
    )

    report = parser._generate_mock_votes_report(dialogue)

    assert report.meeting_title == "Greer City Council Meeting"
    # Should find two mock outcomes (one for park because "park" is absent, but wait!
    # "park" is NOT in dialogue here, so it only generates "budget" and "adjournment" or just "budget"?)
    # Let's check: "park" is not in dialogue, but "budget" is in dialogue.
    # So it should find "budget" outcome.
    outcomes_map = {o.proposal_title: o for o in report.outcomes}
    
    assert "Budget Reconciliation Bill" in outcomes_map
    budget_outcome = outcomes_map["Budget Reconciliation Bill"]
    assert budget_outcome.proposal_id == "ORD-2025-001"
    assert budget_outcome.result == "Passed"
    assert budget_outcome.voting_type == "Voice Vote"
    
    # Assert Smith and Miller voted Aye in voice vote
    votes = {v.councillor_name: v.vote for v in budget_outcome.votes_map}
    assert votes["Davis"] == "Aye"
    assert votes["Miller"] == "Aye"


def test_parse_votes_from_dialogue_api():
    """
    Verifies that parse_votes_from_dialogue executes structured completions
    using OpenAI client Beta parsing, returning the validated model.
    """
    parser = VoteParser()
    dialogue = "[00:01:00] SPEAKER: Roll call."

    # Prepare Mock Report
    mock_vote = IndividualVote(councillor_name="Davis", vote="Aye")
    mock_outcome = VoteOutcome(
        proposal_id="ORD-2025-10",
        proposal_title="Tax rate code",
        result="Passed",
        ayes_count=1,
        nays_count=0,
        abstentions_count=0,
        voting_type="Roll Call",
        votes_map=[mock_vote]
    )
    mock_report = MeetingVotesReport(
        meeting_title="Greer Council",
        outcomes=[mock_outcome]
    )

    # Mock chat completion parse response
    mock_choice = MagicMock()
    mock_choice.message.parsed = mock_report
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    with patch('auditor.vote_parser.OpenAI') as mock_openai_class:
        mock_client = mock_openai_class.return_value
        mock_client.beta.chat.completions.parse.return_value = mock_completion

        # Run
        res_report = parser.parse_votes_from_dialogue(
            dialogue,
            openai_api_key="sk-mockkey"
        )

        # Assertions
        assert res_report is not None
        assert res_report.meeting_title == "Greer Council"
        assert len(res_report.outcomes) == 1
        assert res_report.outcomes[0].proposal_id == "ORD-2025-10"
        assert res_report.outcomes[0].votes_map[0].councillor_name == "Davis"
        assert res_report.outcomes[0].votes_map[0].vote == "Aye"
        
        # Verify call parameters
        mock_client.beta.chat.completions.parse.assert_called_once()
        call_kwargs = mock_client.beta.chat.completions.parse.call_args[1]
        assert call_kwargs["response_format"] == MeetingVotesReport
        assert call_kwargs["model"] == "gpt-4o-mini"
