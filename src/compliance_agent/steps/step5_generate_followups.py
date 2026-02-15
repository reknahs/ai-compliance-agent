
from typing import Dict
from langchain_core.messages import HumanMessage, SystemMessage

from ..state import get_structured_llm, get_llm
from ..schemas.workflow import FollowUpQuestions


def step_5_generate_followups(state: Dict) -> Dict:
    """Step 5: Uses structured output for follow-up questions"""
    print("\n Step 5: Generating Follow-up Questions")
    
    user_query = state["user_query"]
    missing_context = state["missing_context"]
    query_type = state["query_type"]
    unsupported_claims = state.get("unsupported_claims", [])
    
    # don't generate follow-ups if no missing context
    if not missing_context and not unsupported_claims:
        state["follow_up_questions"] = []
        return state
    
    try:
        llm = get_structured_llm(FollowUpQuestions, temperature=0.4)
        
        system_prompt = """You are an expert at asking clarifying questions.

Generate 2-3 specific, actionable follow-up questions that would help provide a more complete answer.

GOOD QUESTIONS:
- "What type of user data will your AI system process?" (specific, actionable)
- "Will this AI system be deployed in the EU?" (relevant for compliance)
- "What is your current model deployment infrastructure?" (contextual)

BAD QUESTIONS:
- "Can you tell me more?" (too vague)
- "What else would you like to know?" (not specific)
- "Do you have any other requirements?" (not actionable)

Make questions concrete and directly related to the missing information."""

        context_parts = []
        if missing_context:
            context_parts.append(f"Missing context: {', '.join(missing_context)}")
        if unsupported_claims:
            context_parts.append(f"Unsupported claims need evidence: {'; '.join(unsupported_claims[:2])}")
        
        user_prompt = f"""USER QUERY: "{user_query}"
QUERY TYPE: {query_type}

GAPS IDENTIFIED:
{chr(10).join(context_parts)}

Generate 2-3 specific follow-up questions that would help address these gaps."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        result: FollowUpQuestions = llm.invoke(messages)
        state["follow_up_questions"] = result.questions
        
    except Exception as e:
        print(f"   Structured follow-up generation failed, using fallback: {e}")
        llm = get_llm(temperature=0.4)
        
        prompt = f"""Generate 2-3 specific follow-up questions for this query: "{user_query}"

Missing context: {missing_context}

Format as numbered list:
1. [Question]
2. [Question]
3. [Question]"""

        response = llm.invoke([HumanMessage(content=prompt)])
        
        follow_ups = []
        for line in response.content.split('\n'):
            line = line.strip()
            if line and line[0].isdigit() and '.' in line:
                question = line.split('.', 1)[1].strip()
                follow_ups.append(question)
        
        state["follow_up_questions"] = follow_ups[:3]
        
    for i, q in enumerate(state["follow_up_questions"], 1):
        print(f"     {i}. {q}")
    
    return state