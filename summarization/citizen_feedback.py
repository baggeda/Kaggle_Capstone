"""
Citizen Feedback and Public Sentiment Analyzer
===============================================
Parses meeting transcripts to extract public comment concerns,
performing sentiment classification and keyword-based clustering of citizen feedback.
"""

import os
from typing import List, Optional
from pydantic import BaseModel, Field
from openai import OpenAI


class CitizenConcern(BaseModel):
    """
    Represents a clustered concern raised by the public during council comment blocks.
    """
    concern_category: str = Field(..., description="General category of the concern (e.g., 'Roads & Infrastructure', 'Zoning & Noise').")
    keywords: List[str] = Field(default_factory=list, description="Keywords clustered from citizen comments (e.g., 'potholes', 'repaving').")
    sentiment: str = Field(..., description="Classified sentiment: 'Supportive', 'Neutral', 'Concerned', 'Frustrated', or 'Hostile'.")
    summary: str = Field(..., description="Brief summary of the public statements and concerns.")
    impact_rating: int = Field(..., description="Intensity and volume rating on a scale of 1 (low) to 5 (high).")


class PublicFeedbackReport(BaseModel):
    """
    Summary report aggregating overall citizen sentiment and specific concern clusters.
    """
    meeting_title: str = Field(..., description="Name or title of the municipal meeting parsed.")
    top_concerns: List[CitizenConcern] = Field(default_factory=list, description="List of major citizen concern clusters.")
    overall_public_sentiment: str = Field(..., description="Overall aggregated mood of the public gallery.")


class CitizenFeedbackAnalyzer:
    """
    Extracts public concerns, classifies sentiment, and clusters keywords from dialogue transcripts.
    """

    def build_analysis_prompt(self, dialogue_text: str) -> str:
        """
        Constructs the prompt for public feedback analysis.
        """
        return (
            "You are an expert public policy analyst and sentiment auditor. Analyze the following meeting dialogue "
            "transcript to isolate the public comment period and extract structured feedback details:\n\n"
            "Analysis Guidelines:\n"
            "1. KEYWORD CLUSTERING: Group similar complaints or support items raised by citizens into distinct "
            "concern categories (e.g. zoning, noise, street paving). Extract key terms they repeat.\n"
            "2. SENTIMENT CLASSIFICATION: Classify the sentiment of the speakers for each concern category. "
            "Label them as 'Supportive', 'Neutral', 'Concerned', 'Frustrated', or 'Hostile'.\n"
            "3. IMPACT RATING: Assign an impact rating from 1 to 5. Consider both the volume (how many citizens "
            "raised the issue) and intensity (how emotionally charged the complaints were).\n"
            "4. OVERALL PUBLIC SENTIMENT: Provide an aggregated assessment of the public gallery's mood.\n\n"
            "Dialogue Transcript:\n"
            "-------------------\n"
            f"{dialogue_text}\n"
        )

    def analyze_feedback(
        self,
        dialogue_text: str,
        openai_api_key: Optional[str] = None
    ) -> PublicFeedbackReport:
        """
        Parses dialogue transcripts to output a structured public feedback report.

        Args:
            dialogue_text: Chronological diarized dialogue transcript.
            openai_api_key: Optional OpenAI API key override.

        Returns:
            A PublicFeedbackReport validation model.
        """
        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")

        if not api_key:
            print("Warning: OpenAI API Key not found. Generating mock public feedback report.")
            return self._generate_mock_feedback_report(dialogue_text)

        prompt = self.build_analysis_prompt(dialogue_text)

        try:
            client = OpenAI(api_key=api_key)
            completion = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a professional public sentiment and concern auditor for municipal proceedings."},
                    {"role": "user", "content": prompt}
                ],
                response_format=PublicFeedbackReport,
                temperature=0.15
            )
            return completion.choices[0].message.parsed

        except Exception as e:
            print(f"Error calling OpenAI structured completions for feedback: {e}. Falling back to mock generator.")
            return self._generate_mock_feedback_report(dialogue_text)

    def _generate_mock_feedback_report(self, dialogue_text: str) -> PublicFeedbackReport:
        """
        Simulates structured public concern records for offline verification.
        """
        concerns = []

        # Simple keyword checks to populate mock clusters
        if "park" in dialogue_text.lower() or "parking" in dialogue_text.lower():
            concerns.append(
                CitizenConcern(
                    concern_category="Zoning & Parking",
                    keywords=["parking", "spaces", "cars", "traffic"],
                    sentiment="Concerned",
                    summary="Citizens raised objections about the lack of parking spaces in the park expansion plans.",
                    impact_rating=4
                )
            )

        if "road" in dialogue_text.lower() or "pothole" in dialogue_text.lower():
            concerns.append(
                CitizenConcern(
                    concern_category="Roads & Infrastructure",
                    keywords=["potholes", "street", "paving", "elm st"],
                    sentiment="Frustrated",
                    summary="Citizens complained about severe potholes on Elm Street requiring repaving.",
                    impact_rating=5
                )
            )

        if not concerns:
            concerns.append(
                CitizenConcern(
                    concern_category="General Municipal Services",
                    keywords=["tax", "trash", "services"],
                    sentiment="Neutral",
                    summary="Citizens requested standard clarifications on local utility tax adjustments.",
                    impact_rating=2
                )
            )

        return PublicFeedbackReport(
            meeting_title="Greer City Council Public Hearing",
            top_concerns=concerns,
            overall_public_sentiment="Divided"
        )
