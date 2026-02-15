import json
from typing import List, Dict
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from pydantic import BaseModel, Field

# Pydantic schemas
class MemoryEvaluationResult(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0, description="Score between 0 and 1")
    reason: str = Field(..., description="Explanation of the score")


class FactExtractionResult(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0, description="F1 score between 0 and 1")
    precision: float = Field(..., ge=0.0, le=1.0, description="Precision score")
    recall: float = Field(..., ge=0.0, le=1.0, description="Recall score")
    reason: str = Field(..., description="Explanation of the scores")




class MemoryRetrievalMetric(BaseMetric):
    def __init__(self, expected_memory_facts: List[str], model):
        self.expected_facts = expected_memory_facts
        self.model = model
        self.score = 0.0
        self.reason = ""
        self.is_successful = False
    
    def measure(self, test_case: LLMTestCase) -> float:
        """measure how well the response uses expected memory facts"""
        
        prompt = f"""You are evaluating if an AI agent correctly retrieved and used stored memory.

List of potential facts that should be mentioned if relevant to the query:
{json.dumps(self.expected_facts, indent=2)}

Actual agent response:
{test_case.actual_output}

Task: Determine what portion of the facts that should have been included are correctly mentioned or implied in the response.

Rules:
- Score 1.0 if ALL expected facts are clearly present
- Score 0.7-0.9 if MOST facts are present
- Score 0.4-0.6 if SOME facts are present
- Score 0.0-0.3 if FEW or NO facts are present
- Facts can be mentioned directly or paraphrased
- Consider semantic similarity, not just exact matches

You must respond with a JSON object matching this exact format:
- score: float between 0.0 and 1.0
- reason: string explaining the score
"""
        
        try:
            result = self.model.generate_structured(prompt, MemoryEvaluationResult)
            self.score = result.score
            self.reason = result.reason
            self.is_successful = True
            
            return self.score
        
        except Exception as e:
            print(f"Error in MemoryRetrievalMetric: {e}")
            self.score = 0.0
            self.reason = f"Evaluation error: {str(e)[:100]}"
            self.is_successful = True
            return 0.0
    
    @property
    def __name__(self):
        return "Memory Retrieval"


class FactExtractionAccuracyMetric(BaseMetric):
    def __init__(self, expected_facts: List[Dict], model):
        self.expected_facts = expected_facts
        self.model = model
        self.score = 0.0
        self.reason = ""
        self.is_successful = False
    
    def measure(self, test_case: LLMTestCase) -> float:
        """Measure accuracy of fact extraction using F1-like scoring"""

        if isinstance(test_case.actual_output, str):
            try:
                output_data = json.loads(test_case.actual_output)
                actual_facts = output_data.get("extracted_facts", [])
            except json.JSONDecodeError:
                actual_facts = []
        elif isinstance(test_case.actual_output, dict):
            actual_facts = test_case.actual_output.get("extracted_facts", [])
        else:
            actual_facts = []
        
        prompt = f"""You are evaluating the accuracy of fact extraction from a conversation.

Expected facts to extract:
{json.dumps(self.expected_facts, indent=2)}

Actually extracted facts:
{json.dumps(actual_facts, indent=2)}

Task: Calculate an F1-style score that considers both precision and recall.

Precision: What percentage of extracted facts are correct?
Recall: What percentage of expected facts were extracted?
F1 Score: 2 * (Precision * Recall) / (Precision + Recall)

Rules:
- Match facts by semantic meaning, not exact strings
- A fact is "correct" if it has the right category, field, and value
- Empty expected facts = score 1.0 if nothing extracted (correct rejection)
- Empty extracted facts when expected = score 0.0 (missed all)

Provide the F1 score, precision, recall (all between 0.0 and 1.0), and a reason explaining your scores.
"""
        
        try:
            result = self.model.generate_structured(prompt, FactExtractionResult)
            self.score = result.score
            self.reason = result.reason
            self.precision = result.precision
            self.recall = result.recall
            self.is_successful = True
            
            return self.score
        
        except Exception as e:
            print(f"Error in FactExtractionAccuracyMetric: {e}")
            self.score = 0.0
            self.reason = f"Evaluation error: {str(e)[:100]}"
            self.precision = 0.0
            self.recall = 0.0
            self.is_successful = True
            return 0.0
    
    @property
    def __name__(self):
        return "Fact Extraction Accuracy"