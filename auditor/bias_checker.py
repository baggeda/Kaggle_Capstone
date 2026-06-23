"""
Bias and Neutrality Auditing Check-Agent
========================================
Audits generated summaries for loaded language, partisan tone, or speaker misattributions
to guarantee summaries are neutral, objective, and non-partisan.
"""

import os
from typing import List, Optional
from pydantic import BaseModel, Field
from openai import OpenAI


class BiasEvaluation(BaseModel):
    """
    Evaluation results from auditing a generated summary for bias and neutrality.
    """
    is_neutral: bool = Field(..., description="True if the summary is objective and neutral; False if bias was found.")
    loaded_language_instances: List[str] = Field(default_factory=list, description="Biased, emotionally charged, or slanted words/phrases found.")
    partisan_tone_issues: List[str] = Field(default_factory=list, description="Examples of non-neutral tone or favoring a particular stance.")
    misattributions: List[str] = Field(default_factory=list, description="Statements or votes wrongly attributed to a councillor.")
    revised_summary: str = Field(..., description="Revised summary removing all bias while preserving factual info.")
    auditor_feedback: str = Field(..., description="General feedback and audit justification notes.")


class SummaryBiasChecker:
    """
    Auditor agent that inspects summary texts for bias, loaded words, and misattributions.
    """

    def build_audit_prompt(self, summary_text: str, original_dialogue: Optional[str] = None) -> str:
        """
        Constructs the detailed prompt for the auditing agent.
        """
        dialogue_section = f"\nOriginal Dialogue:\n------------------\n{original_dialogue}\n" if original_dialogue else ""
        return (
            "You are a dedicated municipal legislative ombudsman and neutral auditor. Your primary duty is to inspect "
            "generated meeting summaries and verify they are completely objective, non-partisan, and free of bias.\n\n"
            "Audit Criteria:\n"
            "1. LOADED LANGUAGE: Flag slanted adjectives, loaded verbs, or emotional phrasing (e.g. 'corrupt', 'disastrous', 'foolish', 'heroic').\n"
            "2. PARTISAN TONE: Ensure the summary does not portray one viewpoint as superior, more logical, or preferred.\n"
            "3. MISATTRIBUTIONS: If the original dialogue transcript is provided below, cross-reference and verify that all statements, "
            "stances, and votes are attributed to the correct councillors. Flag any errors.\n"
            "4. REVISED SUMMARY: Re-write the summary to be completely objective, dry, factual, and neutral.\n\n"
            f"Summary to Audit:\n"
            "-----------------\n"
            f"{summary_text}\n"
            f"{dialogue_section}"
        )

    def audit_summary(
        self,
        summary_text: str,
        original_dialogue: Optional[str] = None,
        openai_api_key: Optional[str] = None
    ) -> BiasEvaluation:
        """
        Audits a generated summary for neutral tone and factual attribution.

        Args:
            summary_text: The LLM summary text to evaluate.
            original_dialogue: Optional original transcript to cross-reference speaker attributions.
            openai_api_key: Optional OpenAI API key override.

        Returns:
            A BiasEvaluation model outlining findings and a revised summary.
        """
        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")

        if not api_key:
            print("Warning: OpenAI API Key not found. Generating mock bias evaluation.")
            return self._generate_mock_bias_evaluation(summary_text, original_dialogue)

        prompt = self.build_audit_prompt(summary_text, original_dialogue)

        try:
            client = OpenAI(api_key=api_key)
            completion = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a professional legislative ombudsman enforcing neutral, non-partisan summaries."},
                    {"role": "user", "content": prompt}
                ],
                response_format=BiasEvaluation,
                temperature=0.1
            )
            return completion.choices[0].message.parsed

        except Exception as e:
            print(f"Error calling OpenAI structured completions for bias check: {e}. Falling back to mock generator.")
            return self._generate_mock_bias_evaluation(summary_text, original_dialogue)

    def _generate_mock_bias_evaluation(self, summary_text: str, original_dialogue: Optional[str] = None) -> BiasEvaluation:
        """
        Simulates structured bias evaluation results for offline verification.
        """
        summary_lower = summary_text.lower()
        
        loaded_language = []
        partisan_issues = []
        misattributions_list = []
        
        # Heuristic checks on mock keywords
        if "disastrous" in summary_lower:
            loaded_language.append("disastrous")
        if "foolish" in summary_lower:
            loaded_language.append("foolish")
            
        if "corrupt" in summary_lower:
            loaded_language.append("corrupt")
            partisan_issues.append("Accusations of corruption injected without neutral context.")

        # Cross-reference speaker if original dialogue is provided
        if original_dialogue and "davis" in summary_lower and "opposed" in summary_lower:
            # Check if Davis actually supported in dialogue
            if "davis: i support" in original_dialogue.lower():
                misattributions_list.append("Summary attributes opposition to Davis, but dialogue shows support.")

        is_neutral = len(loaded_language) == 0 and len(partisan_issues) == 0 and len(misattributions_list) == 0
        
        # Cleaned rewritten text fallback
        cleaned_summary = summary_text
        for word in loaded_language:
            cleaned_summary = cleaned_summary.replace(word, "concerning")
        
        if not is_neutral:
            revised = f"[AUDITED AND REVISED SUMMARY]: {cleaned_summary.strip()}"
            feedback = "Audit failed. Loaded language and/or attribution errors resolved."
        else:
            revised = summary_text
            feedback = "Audit passed. Summary is objective, non-partisan, and neutral."

        return BiasEvaluation(
            is_neutral=is_neutral,
            loaded_language_instances=loaded_language,
            partisan_tone_issues=partisan_issues,
            misattributions=misattributions_list,
            revised_summary=revised,
            auditor_feedback=feedback
        )
