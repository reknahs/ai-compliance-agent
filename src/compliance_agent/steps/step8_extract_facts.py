from typing import Dict, List
from langchain_core.messages import HumanMessage, SystemMessage

from ..state import get_structured_llm
from ..schemas import ExtractedFact
from pydantic import BaseModel, Field


class FactExtractionResult(BaseModel):
    """Result from fact extraction"""
    facts: List[ExtractedFact] = Field(default_factory=list, description="List of extracted facts")


def step_8_extract_facts(state: Dict, memory_manager=None) -> Dict:
    """Step 8: Extract facts from conversation and update user profile"""
    print("\n Step 8: Extracting Facts from Conversation")
    
    # only extract if conversation was approved
    if not state.get("human_approved", False):
        print("     Skipping fact extraction (conversation not approved)")
        state["extracted_facts"] = []
        state["profile_updated"] = False
        state["profile_conflicts"] = []
        return state
    
    if not memory_manager or not hasattr(memory_manager, 'profile_manager'):
        print("   No profile manager available, skipping fact extraction")
        state["extracted_facts"] = []
        state["profile_updated"] = False
        state["profile_conflicts"] = []
        return state
    
    user_query = state["user_query"]
    agent_response = state["structured_answer"]
    
    try:        
        llm = get_structured_llm(FactExtractionResult, temperature=0.2)
        
        system_prompt = """You are an expert at extracting factual information about users from conversations.

Analyze the conversation and extract ONLY information about the user themselves.

CATEGORIES:
1. personal_info - Extract to fields: name, role, company, location, industry
2. preference - Extract user preferences about: response_style, detail_level, tools, methods
3. expertise - Extract domains of expertise with skill levels: beginner, intermediate, advanced, expert

EXTRACTION RULES:
- Extract facts explicitly stated by the USER about themselves
- Do NOT extract information about general topics, third parties, or patients
- Do NOT extract facts from the agent's response (only from user's message)
- Location phrases: "I'm located in X", "I'm in X", "I'm based in X" → personal_info.location
- Company phrases: "I work at X", "I'm working at X", "I'm employed by X" → personal_info.company
- Role phrases: "I'm a X", "I work as a X", "My role is X" → personal_info.role
- Preference phrases: "brief", "detailed", "concise", "I prefer X", "give me X" → preference
- Industry phrases: "I work in X industry", "My industry is X" → personal_info.industry

DETAILED EXAMPLES:

Example 1:
User: "I'm a data scientist at Google"
→ Extract:
  - category: personal_info, field: role, value: "data scientist"
  - category: personal_info, field: company, value: "Google"

Example 2:
User: "Give me a brief overview please"
→ Extract:
  - category: preference, field: detail_level, value: "brief"

Example 3:
User: "I work primarily with Python and TensorFlow"
→ Extract:
  - category: expertise, field: Python, value: "advanced: primary tool"
  - category: expertise, field: TensorFlow, value: "advanced: primary tool"

Example 4:
User: "What is GDPR?"
→ Extract: [] (no personal information)

Example 5:
User: "I'm located in America"
→ Extract:
  - category: personal_info, field: location, value: "America"

Example 6:
User: "I'm working at Apple and they need a brief paragraph"
→ Extract:
  - category: personal_info, field: company, value: "Apple"
  - category: preference, field: detail_level, value: "brief"

Example 7:
User: "I'm a lawyer. I'm located in the USA."
→ Extract:
  - category: personal_info, field: role, value: "lawyer"
  - category: personal_info, field: location, value: "USA"

Return empty list if no personal facts about the user are found."""

        user_prompt = f"""USER MESSAGE: "{user_query}"

Extract any facts about the USER from this conversation. Focus ONLY on what the user explicitly stated about THEMSELVES (not about their patients, clients, or other people).

Return structured facts (or empty list if none found)."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        result: FactExtractionResult = llm.invoke(messages)
        extracted_facts = result.facts
        
    except Exception as e:
        print(f"   Fact extraction failed: {e}")
        extracted_facts = []
    
    # store extracted facts in state
    state["extracted_facts"] = [fact.model_dump() for fact in extracted_facts]
    
    if not extracted_facts:
        print("    No facts extracted from this conversation")
        state["profile_updated"] = False
        state["profile_conflicts"] = []
        return state
    
    print(f"   Extracted {len(extracted_facts)} fact(s):")
    for fact in extracted_facts:
        print(f"      - {fact.category}.{fact.field} = {fact.value}")
    
    # apply facts to profile
    profile_manager = memory_manager.profile_manager
    result = profile_manager.apply_extracted_facts(extracted_facts)
    
    state["profile_updated"] = len(result["updated"]) > 0
    state["profile_conflicts"] = result["conflicts"]
    
    if result["conflicts"]:
        print(f"\n   Detected {len(result['conflicts'])} potential conflict(s) - stored as additional values:")
        for conflict in result["conflicts"]:
            print(f"      - {conflict}")
    
    if result["updated"]:
        print(f"\n   Profile updated with {len(result['updated'])} fact(s):")
        for update in result["updated"][:3]:  # Show first 3
            print(f"      - {update}")
    
    return state