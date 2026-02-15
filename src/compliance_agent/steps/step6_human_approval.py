from typing import Dict


def step_6_human_approval(state: Dict) -> Dict:
    """Displays response and waits for human approval before storing conversation"""
    print("\nStep 6: Human Approval")
    
    final_response = format_response(state)
    
    print("\n" + "=" * 80)
    print("FINAL RESPONSE")
    print("=" * 80 + "\n")
    print(final_response)
    
    state["final_response"] = final_response

    if state.get("auto_approve", True):
        state["human_approved"] = True
        state["human_feedback"] = "Auto-approved (testing mode)"
        print("     Response auto-approved (testing mode)")
        return state
    
    # get human approval
    print("\n" + "=" * 80)
    print("HUMAN APPROVAL REQUIRED")
    print("=" * 80)
    print(f"\nCitation Quality: {state['citation_quality']}")
    print(f"Query Type: {state['query_type']}")
    print("\nDo you approve storing this response in memory?")
    print("Options: 'yes' or 'approve' to approve, anything else to reject")
    print("=" * 80 + "\n")
    
    user_input = input("Your decision: ").strip().lower()
    
    if user_input in ["yes", "y", "approve"]:
        state["human_approved"] = True
        state["human_feedback"] = "Approved"
        print("     Response approved by human")
    else:
        state["human_approved"] = False
        state["human_feedback"] = user_input if user_input else "Rejected"
        print("     Response rejected by human")
    
    return state


def format_response(state: Dict) -> str:
    """Format the final response with quality metrics."""
    final_response = ""
    final_response += "=" * 80 + "\n"
    final_response += "AI COMPLIANCE & SECURITY AGENT - RESPONSE\n"
    final_response += "=" * 80 + "\n\n"
    
    quality = state["citation_quality"]
    quality_emoji = {
        "Excellent": "🟢",
        "Good": "🟢", 
        "Fair": "🟡",
        "Poor": "🔴"
    }
    badge = quality_emoji.get(quality, "⚪")
    
    final_response += f"{badge} **Answer Quality: {quality}**\n\n"
    
    final_response += state["structured_answer"]
    final_response += "\n\n"
    
    retrieval_scores = state.get("retrieval_scores", [])
    if retrieval_scores:
        avg_score = sum(retrieval_scores) / len(retrieval_scores)
        final_response += "---\n\n"
        final_response += "###   Quality Metrics:\n\n"
        final_response += f"- **Citation Quality:** {quality}\n"
        final_response += f"- **Sources Retrieved:** {len(state['retrieved_chunks'])} documents\n"
        final_response += f"- **Average Relevance:** {avg_score:.1%}\n"
        
        if state.get("loop_count", 0) > 0:
            final_response += f"- **Refinement Iterations:** {state['loop_count']}\n"
        
        final_response += "\n"
    
    # follow-up questions
    if state["follow_up_questions"]:
        final_response += "---\n\n"
        final_response += "###   Follow-up Questions for More Specific Guidance:\n\n"
        for i, question in enumerate(state["follow_up_questions"], 1):
            final_response += f"{i}. {question}\n"
        final_response += "\n"
    
    # show unsupported claims if validation flagged them
    unsupported_claims = state.get("unsupported_claims", [])
    if unsupported_claims and quality in ["Fair", "Poor"]:
        final_response += "---\n\n"
        final_response += "###   Claims Needing Additional Evidence:\n\n"
        for claim in unsupported_claims[:3]:
            final_response += f"- {claim}\n"
        final_response += "\n"
    
    # sources used
    sources = set([chunk["source"].split("/")[-1] for chunk in state["retrieved_chunks"]])
    final_response += f"- Based on {len(sources)} source document(s): {', '.join(sorted(sources))}\n"
    
    final_response += "\n" + "=" * 80 + "\n"
    
    return final_response