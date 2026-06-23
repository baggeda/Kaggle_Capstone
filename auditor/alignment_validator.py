"""
Summary Alignment Validator
===========================
Automates verification that generated summaries, alignments, and voting records
map to valid timestamps and speakers in the raw diarized transcript.
"""

import re
from typing import List, Optional, Set
from pydantic import BaseModel, Field

from transcription.audio_processor import DiarizedTranscript
from summarization.agenda_aligner import AlignmentReport
from summarization.summary_engine import StructuredCivicSummary
from auditor.vote_parser import MeetingVotesReport


class ValidationIssue(BaseModel):
    """
    Represents an alignment or attribution error found during auditing.
    """
    issue_type: str = Field(..., description="Category: 'invalid_timestamp', 'out_of_bounds', 'out_of_order', 'unknown_speaker'.")
    description: str = Field(..., description="Detailed description of the validation failure.")
    entity_id: Optional[str] = Field(None, description="The ID of the target agenda item or councillor.")


class ValidationReport(BaseModel):
    """
    Report consolidating all validation results.
    """
    is_valid: bool = Field(..., description="True if no issues are detected; False otherwise.")
    issues: List[ValidationIssue] = Field(default_factory=list, description="Chronological list of validation errors.")


class SummaryAlignmentValidator:
    """
    Validates timestamps and speaker names against the ground-truth diarized transcript.
    """

    def parse_timestamp_to_seconds(self, ts_str: str) -> Optional[float]:
        """
        Parses [HH:MM:SS] or HH:MM:SS formats into floating-point seconds.
        """
        pattern = re.compile(r'\[?(\d{2}):(\d{2}):(\d{2})\]?')
        match = pattern.match(ts_str.strip())
        if not match:
            return None
        
        hours, minutes, seconds = map(int, match.groups())
        return float(hours * 3600 + minutes * 60 + seconds)

    def _get_transcript_speakers(self, raw_transcript: DiarizedTranscript) -> Set[str]:
        """
        Extracts the set of normalized speaker names from the transcript.
        """
        speakers = set()
        for line in raw_transcript.lines:
            speakers.add(line.speaker.upper().strip())
        return speakers

    def _get_transcript_duration(self, raw_transcript: DiarizedTranscript) -> float:
        """
        Determines the total duration of the transcript in seconds.
        """
        if not raw_transcript.lines:
            return 0.0
        
        # Parse the timestamp of the last dialogue line
        last_ts = raw_transcript.lines[-1].timestamp
        last_secs = self.parse_timestamp_to_seconds(last_ts)
        return last_secs or 0.0

    def validate_alignment_report(
        self,
        report: AlignmentReport,
        raw_transcript: DiarizedTranscript
    ) -> ValidationReport:
        """
        Validates timestamps in an AlignmentReport against raw transcript bounds.
        """
        issues = []
        max_duration = self._get_transcript_duration(raw_transcript)

        for segment in report.segments:
            start_secs = self.parse_timestamp_to_seconds(segment.start_timestamp)
            end_secs = self.parse_timestamp_to_seconds(segment.end_timestamp)

            # 1. Syntax check
            if start_secs is None:
                issues.append(
                    ValidationIssue(
                        issue_type="invalid_timestamp",
                        description=f"Start timestamp '{segment.start_timestamp}' is invalid.",
                        entity_id=segment.agenda_item_id
                    )
                )
            if end_secs is None:
                issues.append(
                    ValidationIssue(
                        issue_type="invalid_timestamp",
                        description=f"End timestamp '{segment.end_timestamp}' is invalid.",
                        entity_id=segment.agenda_item_id
                    )
                )

            if start_secs is not None and end_secs is not None:
                # 2. Chronological order check
                if start_secs > end_secs:
                    issues.append(
                        ValidationIssue(
                            issue_type="out_of_order",
                            description=f"Start time ({segment.start_timestamp}) is after end time ({segment.end_timestamp}).",
                            entity_id=segment.agenda_item_id
                        )
                    )
                
                # 3. Temporal boundary check (allow 5s buffer)
                if start_secs < 0.0 or end_secs > max_duration + 5.0:
                    issues.append(
                        ValidationIssue(
                            issue_type="out_of_bounds",
                            description=f"Segment times [{segment.start_timestamp} - {segment.end_timestamp}] exceed transcript range [0.0 - {max_duration}s].",
                            entity_id=segment.agenda_item_id
                        )
                    )

        return ValidationReport(is_valid=len(issues) == 0, issues=issues)

    def validate_structured_summary(
        self,
        summary: StructuredCivicSummary,
        raw_transcript: DiarizedTranscript
    ) -> ValidationReport:
        """
        Validates that speakers in a StructuredCivicSummary exist in the raw transcript.
        """
        issues = []
        valid_speakers = self._get_transcript_speakers(raw_transcript)

        for perspective in summary.councillor_perspectives:
            # Normalize name (e.g. 'Councillor Davis' -> match 'Davis' in transcript speakers)
            name_normalized = perspective.councillor_name.upper().replace("COUNCILLOR", "").strip()
            
            # Check if name is partially contained in the speaker list
            found = False
            for speaker in valid_speakers:
                if name_normalized in speaker or speaker in name_normalized:
                    found = True
                    break

            if not found:
                issues.append(
                    ValidationIssue(
                        issue_type="unknown_speaker",
                        description=f"Councillor '{perspective.councillor_name}' did not speak in the raw transcript.",
                        entity_id=perspective.councillor_name
                    )
                )

        return ValidationReport(is_valid=len(issues) == 0, issues=issues)

    def validate_votes_report(
        self,
        votes_report: MeetingVotesReport,
        raw_transcript: DiarizedTranscript
    ) -> ValidationReport:
        """
        Validates that voters listed in a MeetingVotesReport are present in the transcript.
        """
        issues = []
        valid_speakers = self._get_transcript_speakers(raw_transcript)

        for outcome in votes_report.outcomes:
            for vote_cast in outcome.votes_map:
                name_normalized = vote_cast.councillor_name.upper().strip()
                
                # Absent councillors are allowed to not speak, skip check if vote is 'Absent'
                if vote_cast.vote.lower() == "absent":
                    continue

                found = False
                for speaker in valid_speakers:
                    if name_normalized in speaker or speaker in name_normalized:
                        found = True
                        break

                if not found:
                    issues.append(
                        ValidationIssue(
                            issue_type="unknown_speaker",
                            description=f"Voting councillor '{vote_cast.councillor_name}' is not in the transcript speaker list.",
                            entity_id=vote_cast.councillor_name
                        )
                    )

        return ValidationReport(is_valid=len(issues) == 0, issues=issues)
