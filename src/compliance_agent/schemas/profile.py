from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class PersonalInfo(BaseModel):
    """Personal information about the user."""
    name: Optional[str] = None
    role: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    industry: Optional[str] = None
    last_updated: str = Field(default_factory=lambda: datetime.now().isoformat())


class Preference(BaseModel):
    """User preference with metadata."""
    preference_type: str = Field(description="Type of preference (e.g., 'response_style', 'detail_level')")
    value: str = Field(description="The preference value")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence in this preference")
    last_updated: str = Field(default_factory=lambda: datetime.now().isoformat())
    occurrences: int = Field(default=1, description="Number of times this preference was observed")


class Expertise(BaseModel):
    """Domain expertise with skill level."""
    domain: str = Field(description="Area of expertise (e.g., 'Python', 'Machine Learning')")
    skill_level: str = Field(description="One of: beginner, intermediate, advanced, expert")
    context: Optional[str] = None
    last_updated: str = Field(default_factory=lambda: datetime.now().isoformat())


class UserProfile(BaseModel):
    """Complete user profile."""
    personal_info: PersonalInfo = Field(default_factory=PersonalInfo)
    preferences: List[Preference] = Field(default_factory=list)
    expertise: List[Expertise] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = Field(default_factory=lambda: datetime.now().isoformat())


class ExtractedFact(BaseModel):
    """Structured fact extracted from conversation."""
    category: str = Field(description="One of: personal_info, preference, expertise")
    field: str = Field(description="Specific field name")
    value: str = Field(description="The extracted value")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    source_context: str = Field(description="The conversation snippet this came from")