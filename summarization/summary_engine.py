"""
Summary Engine Module
=====================
Defines classes and schemas for formatting prompts and processing LLM summaries
using raw standard libraries and structured schemas.
"""

import os
from typing import List, Optional
from pydantic import BaseModel, Field
from openai import OpenAI


class SummaryReport(BaseModel):
    """
    Structured summary output representing the key details of a legislative meeting.
    """
    meeting_title: str = Field(..., description="The title or description of the council meeting.")
    executive_summary: str = Field(..., description="A high-level readable overview of the proceedings.")
    key_decisions: List[str] = Field(default_factory=list, description="Major legislative proposals, votes, or policies resolved.")
    public_contributions: List[str] = Field(default_factory=list, description="Public commentary, concerns, or citizen input noted.")
    next_steps: List[str] = Field(default_factory=list, description="Future action items or scheduled reviews.")


class DebatePoint(BaseModel):
    """
    Represents a specific topic debated during the meeting with arguments for and against.
    """
    topic: str = Field(..., description="The core issue or proposal being debated.")
    arguments_for: List[str] = Field(default_factory=list, description="Arguments presented in favor of the proposal.")
    arguments_against: List[str] = Field(default_factory=list, description="Arguments presented in opposition to the proposal.")


class CouncillorPerspective(BaseModel):
    """
    Tracks an individual councillor's arguments, stance, and any vocal opposition.
    """
    councillor_name: str = Field(..., description="Name or identifier of the councillor.")
    stance: str = Field(..., description="The councillor's position (e.g. Support, Oppose, Neutral, Undecided).")
    key_arguments: List[str] = Field(default_factory=list, description="Key points or arguments raised by this councillor.")
    vocal_opposition: bool = Field(..., description="Flag indicating if the councillor was vocal against the topic at hand.")


class AudienceResponse(BaseModel):
    """
    Captures general public sentiment, cries, objections, or support noted in the audience.
    """
    general_sentiment: str = Field(..., description="Overall mood/sentiment of the public gallery (e.g. Hostile, Supportive, Mixed, Attentive).")
    public_speaker_concerns: List[str] = Field(default_factory=list, description="Specific issues raised by citizens during public comment blocks.")
    objections_or_applause: List[str] = Field(default_factory=list, description="Recorded instances of audience outcry, objections, applause, or disruption.")


class StructuredCivicSummary(BaseModel):
    """
    Detailed, LLM-generated meeting summary structuring debates, councillor stances, and public sentiment.
    """
    agenda_item_title: str = Field(..., description="Title of the agenda item being summarized.")
    debate_points: List[DebatePoint] = Field(default_factory=list, description="List of key debate topics extracted.")
    councillor_perspectives: List[CouncillorPerspective] = Field(default_factory=list, description="Individual councillor opinions and opposition flags.")
    audience_response: AudienceResponse = Field(..., description="Audience reactions and citizen concern details.")
    summary_narrative: str = Field(..., description="High-level descriptive overview summarizing the discussion flow.")


class SummaryEngine:
    """
    Coordinates prompt construction and API invocation with LLM providers (e.g. OpenAI)
    to process documents and transcripts.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initializes the SummaryEngine, optionally with an OpenAI API key.
        """
        # If API key is provided, initialize client; otherwise, it remains a mock or uses env vars.
        self.api_key = api_key
        self._client = OpenAI(api_key=api_key) if api_key else None

    def build_prompt(self, document_content: str) -> str:
        """
        Constructs a structured prompt guide for the LLM to format local proceedings.

        Args:
            document_content: The raw text representation of the meeting transcript or proposals.

        Returns:
            A formatted prompt string.
        """
        return (
            "You are an expert civic engagement agent. Analyze the following meeting text "
            "and extract a structured summary report containing: meeting title, "
            "executive summary, key decisions, public contributions, and next steps.\n\n"
            f"Meeting Text:\n{document_content}\n"
        )

    def build_structured_prompt(self, dialogue_text: str, agenda_item_title: str) -> str:
        """
        Builds a repeatable prompt instructing the LLM to extract key debate points,
        councillor perspectives (focusing on vocal opposition), and audience sentiment.
        """
        return (
            "You are a professional civic engagement analyst monitoring local government proceedings. "
            f"Analyze the meeting dialogue stream regarding the agenda topic: '{agenda_item_title}'.\n\n"
            "Analyze and extract the following structured details:\n"
            "1. KEY DEBATE POINTS: Extract the core arguments presented. Document both the arguments in favor "
            "and the arguments against the proposal.\n"
            "2. COUNCILLOR PERSPECTIVES: Identify each councillor by name, their stance, and key arguments. "
            "Identify and explicitly flag any councillors who are vocal against (in opposition to) the topic at hand.\n"
            "3. AUDIENCE SENTIMENT AND RESPONSE: Assess the overall public sentiment, extract specific concerns "
            "voiced by citizens/public speakers, and record any objections, shouts, applause, or disruptions.\n"
            "4. SUMMARY NARRATIVE: Provide a concise narrative of the debate flow.\n\n"
            "Meeting Dialogue:\n"
            "-----------------\n"
            f"{dialogue_text}\n"
        )

    def generate_summary(self, document_content: str) -> SummaryReport:
        """
        Invokes the LLM pipeline or uses a deterministic fallback to produce a structured summary.
        """
        prompt = self.build_prompt(document_content)

        if not self._client:
            # Fallback mock summary for offline testing and verification
            return SummaryReport(
                meeting_title="Mock City Council Meeting Summary",
                executive_summary="This is a mock overview of the meeting, indicating offline mode.",
                key_decisions=["Approved the park renovation proposal.", "Deferred the tax hearing."],
                public_contributions=["Citizen complained about potholes on Elm St."],
                next_steps=["Schedule follow-up on July 10th."]
            )

        response = self._client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful civic assistant parsing meeting records."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        raw_text = response.choices[0].message.content or ""
        return SummaryReport(
            meeting_title="Extracted City Council Meeting",
            executive_summary=raw_text[:200] + "...",
            key_decisions=["Extracted from LLM content"],
            public_contributions=[],
            next_steps=[]
        )

    def generate_structured_summary(
        self,
        dialogue_text: str,
        agenda_item_title: str,
        openai_api_key: Optional[str] = None
    ) -> StructuredCivicSummary:
        """
        Invokes OpenAI model to extract a structured civic summary detailing debates,
        councillor perspectives (focusing on opposition), and public sentiment.
        """
        api_key = openai_api_key or self.api_key or os.environ.get("OPENAI_API_KEY")

        if not api_key:
            print("Warning: OpenAI API Key not found. Generating mock structured civic summary.")
            return self._generate_mock_structured_summary(agenda_item_title)

        prompt = self.build_structured_prompt(dialogue_text, agenda_item_title)

        try:
            client = self._client or OpenAI(api_key=api_key)
            completion = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert civic analyst compiling structured council debate reviews."},
                    {"role": "user", "content": prompt}
                ],
                response_format=StructuredCivicSummary,
                temperature=0.15
            )
            return completion.choices[0].message.parsed

        except Exception as e:
            print(f"Error calling OpenAI structured completions: {e}. Falling back to mock generator.")
            return self._generate_mock_structured_summary(agenda_item_title)

    def _generate_mock_structured_summary(self, agenda_item_title: str) -> StructuredCivicSummary:
        """
        Simulates detailed structured summaries for offline tests.
        """
        # Mock Debate Points
        debate_points = [
            DebatePoint(
                topic="Funding Allocation",
                arguments_for=["Promotes community engagement and health.", "Increases property values nearby."],
                arguments_against=["High initial tax expenditure.", "Limits commercial space."]
            )
        ]

        # Mock Councillor Perspectives (with focused opposition)
        councillor_perspectives = [
            CouncillorPerspective(
                councillor_name="Councillor Davis",
                stance="Support",
                key_arguments=["Greer needs green spaces for families."],
                vocal_opposition=False
            ),
            CouncillorPerspective(
                councillor_name="Councillor Miller",
                stance="Oppose",
                key_arguments=["The project is too expensive and lacks a clear parking plan."],
                vocal_opposition=True
            )
        ]

        # Mock Audience Response
        audience_response = AudienceResponse(
            general_sentiment="Mixed",
            public_speaker_concerns=["Noise during construction.", "Impact on traffic flows."],
            objections_or_applause=["Audience groaned when Miller voiced concern.", "Some shouts of 'agree!' from gallery."]
        )

        return StructuredCivicSummary(
            agenda_item_title=agenda_item_title,
            debate_points=debate_points,
            councillor_perspectives=councillor_perspectives,
            audience_response=audience_response,
            summary_narrative="Debate occurred between Davis, who supported the expansion, and Miller, who strongly opposed it."
        )
