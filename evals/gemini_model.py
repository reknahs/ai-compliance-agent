import time
import os
from deepeval.models.base_model import DeepEvalBaseLLM
from google import genai
from google.genai import types
from typing import Type, TypeVar
from pydantic import BaseModel
from dotenv import load_dotenv
import json

# load environment variables
load_dotenv()

T = TypeVar('T', bound=BaseModel)

class GeminiModel(DeepEvalBaseLLM):    
    def __init__(self, model_name="gemma-3-27b-it", api_key=None):
        self.model_name = model_name
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.last_request_time = 0

    def load_model(self):
        pass

    def _rate_limit_wait(self):
        """ to enforce 30 RPM (approx 1 req / 2s)"""
        time_since_last = time.time() - self.last_request_time
        if time_since_last < 2.2:
            sleep_time = 2.2 - time_since_last
            time.sleep(sleep_time)

    def generate(self, prompt: str) -> str:
        """Generate response with rate limiting"""
        self._rate_limit_wait()

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                self.last_request_time = time.time()
                return response.text
            except Exception as e:
                if "429" in str(e) or "ResourceExhausted" in str(e):
                    print("Hit rate limit. Retrying in 10s")
                    time.sleep(10)
                else:
                    print(f"Error: {e}")
                    return ""
        return ""
    
    def generate_structured(self, prompt: str, schema: Type[T]) -> T:
        """Generate structured output using Pydantic schema with Google Genai"""
        self._rate_limit_wait()

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema 
                    )
                )
                self.last_request_time = time.time()
                
                return response.parsed
                
            except Exception as e:
                if "429" in str(e) or "ResourceExhausted" in str(e):
                    print("Hit rate limit. Retrying in 10s")
                    time.sleep(10)
                else:
                    print(f"Error: {e}")
                    return ""
        
        return None

    async def a_generate(self, prompt: str) -> str:
        """Async generate (calls sync version)"""
        return self.generate(prompt)

    def get_model_name(self):
        """Get model name"""
        return self.model_name