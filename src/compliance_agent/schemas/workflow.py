from pydantic import BaseModel, Field
from typing import List
from .profile import ExtractedFact

class IntentAnalysis(BaseModel):
    """Structured intent analysis output."""
    intent_analysis: str = Field(description="Brief analysis of what the user is asking")
    query_type: str = Field(description="One of: security_risk, compliance, comparison, definition, general")
    missing_context: List[str] = Field(description="List of missing information needed, or empty list if sufficient")


class ValidationResult(BaseModel):
    """Structured validation output."""
    citation_quality: str = Field(description="One of: Excellent, Good, Fair, Poor")
    validation_notes: str = Field(description="Brief notes on quality and concerns")
    missing_information: str = Field(description="Specific missing context or documents needed, or 'None'")
    unsupported_claims: List[str] = Field(default_factory=list, description="List of claims lacking citations")


class FollowUpQuestions(BaseModel):
    """Structured follow-up questions output."""
    questions: List[str] = Field(description="List of 2-3 specific follow-up questions")


class HumanApprovalDecision(BaseModel):
    """Structured human approval decision."""
    approved: bool = Field(description="Whether the response is approved for storage")

class FactExtractionResult(BaseModel):
    """Structured output for fact extraction."""
    facts: List[ExtractedFact] = Field(default_factory=list, description="List of extracted facts")