import os
import json
from typing import Dict, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field


from .schemas.profile import ExtractedFact, PersonalInfo, Preference, Expertise, UserProfile


class UserProfileManager:  
    def __init__(self, profile_path: str = "user_profile.json"):
        self.profile_path = profile_path
        
        # ensure directory exists
        directory = os.path.dirname(self.profile_path)
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
            except Exception as e:
                print(f"     Warning: Could not create directory {directory}: {e}")
                
        self.profile = self._load_profile()
    
    def _load_profile(self) -> UserProfile:
        """Load profile from JSON file or create new one"""
        if os.path.exists(self.profile_path):
            try:
                with open(self.profile_path, 'r') as f:
                    data = json.load(f)
                return UserProfile(**data)
            except Exception as e:
                print(f"     Error loading profile: {e}, creating new profile")
                return UserProfile()
        else:
            return UserProfile()
    
    def _save_profile(self):
        """Save profile to JSON file"""
        try:
            self.profile.last_updated = datetime.now().isoformat()
            with open(self.profile_path, 'w') as f:
                json.dump(self.profile.model_dump(), f, indent=2)
        except Exception as e:
            print(f"     Error saving profile: {e}")
    
    def add_personal_info_value(self, field: str, value: str) -> bool:
        """Add a value to personal info field (stored as list)"""
        if not hasattr(self.profile.personal_info, field):
            return False
        
        current_values = getattr(self.profile.personal_info, field, None) or []
        
        value_lower = value.lower()
        if not any(v.lower() == value_lower for v in current_values):
            current_values.append(value)
            self.profile.personal_info.last_updated = datetime.now().isoformat()
            self._save_profile()
            return True
        
        return False
    
    def add_or_update_preference(self, preference_type: str, value: str, confidence: float = 0.5):
        """Add new preference or update existing one"""
        for pref in self.profile.preferences:
            if pref.preference_type == preference_type:
                pref.value = value
                pref.confidence = min(1.0, pref.confidence + 0.1)  # Increase confidence
                pref.occurrences += 1
                pref.last_updated = datetime.now().isoformat()
                self._save_profile()
                return
        
        new_pref = Preference(
            preference_type=preference_type,
            value=value,
            confidence=confidence
        )
        self.profile.preferences.append(new_pref)
        self._save_profile()
    
    def add_or_update_expertise(self, domain: str, skill_level: str, context: Optional[str] = None):
        """Add new expertise or update existing one"""
        for exp in self.profile.expertise:
            if exp.domain.lower() == domain.lower():
                exp.skill_level = skill_level
                if context:
                    exp.context = context
                exp.last_updated = datetime.now().isoformat()
                self._save_profile()
                return
        
        new_exp = Expertise(
            domain=domain,
            skill_level=skill_level,
            context=context
        )
        self.profile.expertise.append(new_exp)
        self._save_profile()
    
    def apply_extracted_facts(self, facts: List[ExtractedFact]) -> Dict[str, List[str]]:
        """Apply extracted facts to profile"""
        updated = []
        conflicts_detected = []
        
        for fact in facts:
            if fact.category == "personal_info":
                current_values = getattr(self.profile.personal_info, fact.field, [])
                
                value_lower = fact.value.lower()
                is_new = not any(v.lower() == value_lower for v in current_values or [])
                
                if current_values and is_new:
                    conflicts_detected.append(
                        f"{fact.field}: existing {current_values} + new '{fact.value}'"
                    )
                
                added = self.add_personal_info_value(fact.field, fact.value)
                if added:
                    updated.append(f"personal_info.{fact.field} += {fact.value}")
            
            elif fact.category == "preference":
                self.add_or_update_preference(fact.field, fact.value, fact.confidence)
                updated.append(f"preference: {fact.field} = {fact.value}")
            
            elif fact.category == "expertise":
                parts = fact.value.split(":", 1)
                if len(parts) == 2:
                    skill_level, context = parts[0].strip(), parts[1].strip()
                else:
                    skill_level, context = "intermediate", fact.value
                
                self.add_or_update_expertise(fact.field, skill_level, context)
                updated.append(f"expertise: {fact.field} ({skill_level})")
        
        return {"updated": updated, "conflicts": conflicts_detected}
    
    def get_formatted_profile(self) -> str:
        """Get formatted profile text for LLM prompts"""
        lines = []
        
        personal_info_lines = []
        if self.profile.personal_info.name:
            personal_info_lines.append(f"  - Name: {', '.join(self.profile.personal_info.name)}")
        if self.profile.personal_info.role:
            personal_info_lines.append(f"  - Role: {', '.join(self.profile.personal_info.role)}")
        if self.profile.personal_info.company:
            personal_info_lines.append(f"  - Company: {', '.join(self.profile.personal_info.company)}")
        if self.profile.personal_info.location:
            personal_info_lines.append(f"  - Location: {', '.join(self.profile.personal_info.location)}")
        if self.profile.personal_info.industry:
            personal_info_lines.append(f"  - Industry: {', '.join(self.profile.personal_info.industry)}")
        
        if personal_info_lines:
            lines.append("**Personal Information:**")
            lines.extend(personal_info_lines)
            lines.append("")
        
        if self.profile.preferences:
            lines.append("**Preferences:**")
            for pref in sorted(self.profile.preferences, key=lambda x: x.confidence, reverse=True):
                lines.append(f"  - {pref.preference_type}: {pref.value} (confidence: {pref.confidence:.2f})")
            lines.append("")
        
        if self.profile.expertise:
            lines.append("**Expertise:**")
            for exp in sorted(self.profile.expertise, key=lambda x: x.domain):
                context_str = f" - {exp.context}" if exp.context else ""
                lines.append(f"  - {exp.domain}: {exp.skill_level}{context_str}")
            lines.append("")
        
        if not lines:
            return "No user profile information available yet."
        
        return "\n".join(lines)
    
    def get_relevant_profile_info(self, query: str) -> str:
        """Extract entities from query and return matching profile sections"""
        query_lower = query.lower()
        relevant = []
        
        # personal info
        if any(word in query_lower for word in ["my", "i", "me", "who", "what do you know"]):
            personal = self.get_formatted_profile()
            if personal != "No user profile information available yet.":
                return personal
        
        # expertise mentions
        for exp in self.profile.expertise:
            if exp.domain.lower() in query_lower:
                relevant.append(f"User expertise in {exp.domain}: {exp.skill_level}")
        
        # preference-related queries
        if any(word in query_lower for word in ["brief", "detailed", "summary", "explain"]):
            for pref in self.profile.preferences:
                if pref.preference_type in ["response_style", "detail_level"]:
                    relevant.append(f"User prefers {pref.preference_type}: {pref.value}")
        
        return "\n".join(relevant) if relevant else self.get_formatted_profile()
    
    def clear_profile(self):
        """Clear all profile data"""
        self.profile = UserProfile()
        self._save_profile()
        if os.path.exists(self.profile_path):
            os.remove(self.profile_path)
        print("     User profile cleared")
    
    def get_profile_stats(self) -> Dict:
        """Get statistics about the profile"""
        return {
            "has_personal_info": any([
                self.profile.personal_info.name,
                self.profile.personal_info.role,
                self.profile.personal_info.company
            ]),
            "preference_count": len(self.profile.preferences),
            "expertise_count": len(self.profile.expertise),
            "created_at": self.profile.created_at,
            "last_updated": self.profile.last_updated
        }