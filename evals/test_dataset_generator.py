from typing import List, Dict
from deepeval.dataset import Golden


class TestDatasetGenerator:    
    @staticmethod
    def generate_rag_test_cases() -> List[Golden]:
        """Generate RAG test cases covering different query types"""
        return [
            # compliance query
            Golden(
                input="What are the requirements for high-risk AI systems under EU AI Act?",
                expected_output="The EU AI Act establishes requirements for high-risk AI systems including conformity assessments, technical documentation, risk management systems, and transparency obligations.",
            ),
            
            # comparison query
            Golden(
                input="Compare NIST AI RMF and EU AI Act approaches to AI governance",
                expected_output="NIST AI RMF provides a voluntary risk management framework focused on trustworthy AI development, while EU AI Act is legally binding regulation with risk-based classification and mandatory requirements.",
            ),
            
            # security risk query
            Golden(
                input="What security risks should I consider for AI chatbots?",
                expected_output="AI chatbots face security risks including prompt injection attacks, data leakage through training data exposure, adversarial inputs, and unauthorized access to sensitive information.",
            ),
            
            # definition query
            Golden(
                input="What is the NIST AI Risk Management Framework?",
                expected_output="The NIST AI Risk Management Framework is a voluntary framework that provides guidance for managing risks associated with AI systems throughout their lifecycle.",
            ),
            
            # technical guidance query
            Golden(
                input="What documentation is required for medical AI systems under EU AI Act?",
                expected_output="Medical AI systems classified as high-risk require technical documentation including system design specifications, data governance procedures, risk assessments, validation reports, and performance metrics.",
            ),
        ]
    
    @staticmethod
    def generate_memory_test_cases() -> List[Dict]:
        """memory retrieval test cases with setup + query"""
        return [
            {
                "test_id": "memory_001",
                "setup_messages": [
                    "I'm a compliance officer at a healthcare company in California"
                ],
                "query": "What's my role and where do I work?",
                "expected_facts": ["compliance officer", "healthcare company", "California"],
                "fact_type": "personal_info"
            },
            {
                "test_id": "memory_002",
                "setup_messages": [
                    "I prefer brief summaries with bullet points"
                ],
                "query": "Tell me about the EU AI Act",
                "expected_facts": ["brief", "summaries", "bullet points"],
                "fact_type": "preferences"
            },
            {
                "test_id": "memory_003",
                "setup_messages": [
                    "I'm working on implementing AI governance for medical devices",
                    "My team is evaluating NIST AI RMF"
                ],
                "query": "What do you know about my current project?",
                "expected_facts": ["AI governance", "medical devices", "NIST AI RMF"],
                "fact_type": "expertise"
            }
        ]
    
    @staticmethod
    def generate_fact_extraction_test_cases() -> List[Dict]:
        """test fact extraction accuracy."""
        return [
            {
                "test_id": "fact_001",
                "input": "I'm a data scientist at Google in Mountain View",
                "expected_facts": [
                    {"category": "personal_info", "field": "role", "value": "data scientist"},
                    {"category": "personal_info", "field": "company", "value": "Google"},
                    {"category": "personal_info", "field": "location", "value": "Mountain View"}
                ]
            },
            {
                "test_id": "fact_002",
                "input": "My colleague works at Microsoft", 
                "expected_facts": []
            },
            {
                "test_id": "fact_003",
                "input": "I need detailed technical explanations with code examples",
                "expected_facts": [
                    {"category": "preference", "field": "detail_level", "value": "detailed"},
                    {"category": "preference", "field": "response_style", "value": "technical explanations"},
                ]
            }
        ]