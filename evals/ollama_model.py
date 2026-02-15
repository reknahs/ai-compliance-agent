import time
import json
from deepeval.models.base_model import DeepEvalBaseLLM
from typing import Type, TypeVar
from pydantic import BaseModel
from langchain_ollama import ChatOllama

T = TypeVar('T', bound=BaseModel)

class OllamaEvalModel(DeepEvalBaseLLM):    
    def __init__(self, model_name="llama3:8b", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.client = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=0.0,
            num_ctx=8192,
            format="json"
        )
        
    def load_model(self):
        pass

    def generate(self, prompt: str) -> str:
        """Generate text response"""
        try:
            response = self.client.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"Error generating response: {e}")
            return ""
    
    def generate_structured(self, prompt: str, schema: Type[T]) -> T:
        """Generate structured output using Pydantic schema with Ollama"""
        
        structured_llm = self.client.with_structured_output(schema)
        
        enhanced_prompt = f"""{prompt}

CRITICAL INSTRUCTIONS:
- You MUST respond with a valid JSON object matching the schema
- Do NOT include any preamble, explanation, or markdown
- Start with {{ and end with }}
- All fields are required
"""
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = structured_llm.invoke(enhanced_prompt)
                
                if isinstance(result, BaseModel):
                    return result
                
                if isinstance(result, dict):
                    return schema(**result)
                
                if isinstance(result, str):
                    result_clean = result.strip()
                    if result_clean.startswith("```"):
                        result_clean = result_clean.split("```")[1]
                        if result_clean.startswith("json"):
                            result_clean = result_clean[4:]
                    
                    json_data = json.loads(result_clean)
                    return schema(**json_data)
                
                print(f"Unexpected result type: {type(result)}")
                return None
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(1)
                
            except Exception as e:
                print(f"Error generating structured output (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(1)
        
        return None

    async def a_generate(self, prompt: str) -> str:
        """Async generate (calls sync version)"""
        return self.generate(prompt)

    def get_model_name(self):
        """Get model name"""
        return self.model_name