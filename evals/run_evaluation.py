import os
import time
import json
from datetime import datetime
from typing import Dict, List
from pathlib import Path

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, HallucinationMetric
from deepeval.test_case import LLMTestCase

from .gemini_model import GeminiModel
from .ollama_model import OllamaEvalModel
from .test_dataset_generator import TestDatasetGenerator
from .custom_metrics import MemoryRetrievalMetric, FactExtractionAccuracyMetric
from src.compliance_agent.main import query_agent, get_memory_manager
import src.compliance_agent.main as main_app
from src.compliance_agent.memory import Mem0MemoryManager, CustomMemoryManager


class EvaluationRunner:  
    def __init__(self, output_dir: str = "data/evaluation_results", ollama_model: str = "llama3:8b", use_custom_memory: bool = False):
        self.output_dir = output_dir
        self.use_custom_memory = use_custom_memory
        os.makedirs(output_dir, exist_ok=True)
        
        self.eval_model = GeminiModel()
        self.ollama_model = OllamaEvalModel(model_name=ollama_model)
        self.current_run_file = None
        self.results = []
        
        # clear memory before starting
        self.clear_all_memory()

    def clear_all_memory(self):
        """Clear both Custom and Mem0 memory systems"""
        print(f"\nClearing ALL Memory (Custom + Mem0)")
        
        # clear custom memory
        try:
            custom_mgr = CustomMemoryManager(
                memory_db_path=os.path.join("data", "memory_store"),
                profile_path=os.path.join("data", "memory_store", "user_profile.json")
            )
            custom_mgr.clear_semantic_memory()
            custom_mgr.profile_manager.clear_profile()
            print("     Custom memory cleared")
        except Exception as e:
            print(f"     Failed to clear Custom memory: {e}")

        # clear Mem0 memory
        try:
            api_key = os.getenv("MEMORY_API_KEY")
            if api_key:
                mem0_mgr = Mem0MemoryManager(
                    api_key=api_key,
                    short_term_size=10,
                    profile_path=os.path.join("data", "memory_store", "user_profile.json")
                )
                mem0_mgr.clear_semantic_memory()
                mem0_mgr.profile_manager.clear_profile()
                print("     Mem0 memory cleared")
            else:
                 print("     Skipping Mem0 clear (no API key)")
        except Exception as e:
            print(f"     Failed to clear Mem0 memory: {e}")

        main_app.MEMORY_MANAGER = None
    
    def get_latest_run_file(self) -> str:
        """Get the most recent evaluation run file"""
        files = list(Path(self.output_dir).glob("eval_*.txt"))
        if not files:
            return None
        return str(max(files, key=os.path.getctime))
    
    def load_existing_run(self, filepath: str) -> Dict[str, List[Dict]]: 
        """Load results from existing run file, grouped by test type"""
        results = {
            "rag": [],
            "memory": [],
            "fact_extraction": []
        }
        
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith("TEST_CASE:"):
                    test_result = json.loads(line.replace("TEST_CASE:", ""))
                    
                    test_id = test_result.get("test_id", "")
                    
                    if test_id.startswith("memory_"):
                        results["memory"].append(test_result)
                    elif test_id.startswith("fact_"):
                        results["fact_extraction"].append(test_result)
                    elif "relevancy_score" in test_result: 
                        results["rag"].append(test_result)
                    
        return results
    
    def save_test_case_result(self, result: Dict):
        """Append test case result to file immediately"""
        with open(self.current_run_file, 'a') as f:
            f.write(f"TEST_CASE:{json.dumps(result)}\n")
    
    def prompt_user_for_mode(self) -> str:
        """Ask user whether to create new run or resume existing"""
        latest = self.get_latest_run_file()
        
        print("=" * 80)
        print("EVALUATION MODE SELECTION")
        print("=" * 80)
        
        if latest:
            print(f"Found existing run: {latest}")
            existing_results = self.load_existing_run(latest)
            
            print(f"\nCompleted test cases:")
            print(f"    RAG Quality: {len(existing_results['rag'])}")
            print(f"    Memory Retrieval: {len(existing_results['memory'])}")
            print(f"    Fact Extraction: {len(existing_results['fact_extraction'])}")
            print(f"    Total: {sum(len(v) for v in existing_results.values())}")
            
            print("\nOptions:")
            print(" [1] Create NEW evaluation run")
            print(" [2] RESUME existing run (continue incomplete tests)")
        else:
            print("\nNo existing runs found.")
            print("\nOptions:")
            print(" [1] Create NEW evaluation run")
        
        while True:
            choice = input("\nYour choice: ").strip()
            if choice == "1":
                return "new"
            elif choice == "2" and latest:
                return "resume"
            else:
                print("Invalid Choice")
    
    def create_new_run(self):
        """Create a new evaluation run file."""    
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_run_file = f"{self.output_dir}/eval_{timestamp}.txt"
        
        with open(self.current_run_file, 'w') as f:
            f.write(f"EVALUATION RUN: {timestamp}\n")
            f.write(f"STARTED: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
        
        print(f"\nCreated new run: {self.current_run_file}\n")
        return {"rag": [], "memory": [], "fact_extraction": []}
    
    def resume_existing_run(self):
        """Resume an existing evaluation run"""    
        latest = self.get_latest_run_file()
        self.current_run_file = latest
        
        existing_results = self.load_existing_run(latest)
        total = sum(len(v) for v in existing_results.values())
        
        print(f"\nResuming run: {self.current_run_file}")
        print(f"    Already completed: {total} test cases")
        print(f"        RAG: {len(existing_results['rag'])}")
        print(f"        Memory: {len(existing_results['memory'])}")
        print(f"        Fact Extraction: {len(existing_results['fact_extraction'])}\n")
        
        return existing_results
    
    def get_completed_test_ids(self, results: Dict[str, List[Dict]], test_type: str) -> set:
        """Get set of test IDs that have been completed for a specific test type"""
        if test_type == "rag":
            return {r["input"] for r in results["rag"] if "input" in r}
        else:
            return {r["test_id"] for r in results[test_type] if "test_id" in r}
    
    def run_evaluation(self):
        """Run multi-type evaluation (RAG + Memory + Fact Extraction)"""

        mode = self.prompt_user_for_mode()
        
        if mode == "new":
            existing_results = self.create_new_run()
        else:
            existing_results = self.resume_existing_run()
        
        self.evaluate_rag_quality(existing_results)
        self.evaluate_memory_system(existing_results)
        self.clear_all_memory()
        self.evaluate_fact_extraction(existing_results)
        
        self.print_summary()
    
    def evaluate_rag_quality(self, existing_results: Dict[str, List[Dict]]):
        """Evaluate RAG quality metrics"""
        test_cases = TestDatasetGenerator.generate_rag_test_cases()
        completed_inputs = self.get_completed_test_ids(existing_results, "rag")

        print("RUNNING RAG QUALITY EVALUATION")

        for i, golden in enumerate(test_cases, 1):
            # skip if already completed
            if golden.input in completed_inputs:
                print(f"[{i}/{len(test_cases)}] Skipping (already completed): {golden.input[:60]}")
                continue
            
            print(f"\n[{i}/{len(test_cases)}] Testing: {golden.input[:60]}")
            
            try:
                start_time = time.time()
                result = query_agent(
                    golden.input, 
                    verbose=False, 
                    auto_approve=True, 
                    skip_memory=True,
                    use_custom_memory=self.use_custom_memory
                )
                latency = time.time() - start_time
                
                retrieved_data = [c["content"] for c in result.get("retrieved_chunks", [])]

                test_case = LLMTestCase(
                    input=golden.input,
                    actual_output=result["response"],
                    expected_output=golden.expected_output,
                    retrieval_context=retrieved_data,
                    context=retrieved_data
                )
                
                print(f"Evaluating metrics")
                
                relevancy_metric = AnswerRelevancyMetric(model=self.eval_model)
                relevancy_score = relevancy_metric.measure(test_case)
                
                faithfulness_metric = FaithfulnessMetric(model=self.eval_model)
                faithfulness_score = faithfulness_metric.measure(test_case)

                hallucination_metric = HallucinationMetric(model=self.eval_model)
                hallucination_score = hallucination_metric.measure(test_case)
                
                citation_count = result["response"].count("[Source:")
                
                test_result = {
                    "input": golden.input,
                    "query_type": result.get("query_type", "unknown"),
                    "latency": round(latency, 2),
                    "citation_count": citation_count,
                    "relevancy_score": round(relevancy_score, 3),
                    "faithfulness_score": round(faithfulness_score, 3),
                    "hallucination_score": round(hallucination_score, 3),
                    "loop_count": result.get("loop_count", 0),
                    "response_length": len(result["response"]),
                    "timestamp": datetime.now().isoformat()
                }
                
                self.save_test_case_result(test_result)
                
                print(f"Completed:")
                print(f"    Relevancy: {relevancy_score:.3f}")
                print(f"    Faithfulness: {faithfulness_score:.3f}")
                print(f"    Citations: {citation_count}")
                print(f"    Latency: {latency:.2f}s")
                
            except Exception as e:
                print(f"Error: {e}")
                error_result = {
                    "input": golden.input,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                self.save_test_case_result(error_result)
    
    def evaluate_memory_system(self, existing_results: Dict[str, List[Dict]]):
        """Evaluate memory retrieval using Mem0"""
        test_cases = TestDatasetGenerator.generate_memory_test_cases()
        completed_ids = self.get_completed_test_ids(existing_results, "memory")
        
        print("RUNNING MEMORY RETRIEVAL EVALUATION")
        
        for i, test in enumerate(test_cases, 1):
            test_id = test["test_id"]
            
            # skip if already completed
            if test_id in completed_ids:
                print(f"[{i}/{len(test_cases)}] Skipping (already completed): {test_id}")
                continue
            
            print(f"\n[{i}/{len(test_cases)}] Testing: {test_id}")
            
            try:
                self.clear_all_memory()
                
                # setup: store facts in memory
                for setup_msg in test["setup_messages"]:
                    print(f"    Storing: {setup_msg[:60]}")
                    query_agent(setup_msg, verbose=False, auto_approve=True, skip_memory=False, use_custom_memory=self.use_custom_memory)
                
                # wait for Mem0 processing
                print("   Waiting for Mem0 processing")
                time.sleep(6)
                
                memory_manager = get_memory_manager(use_custom_memory=self.use_custom_memory)
                memory_manager.clear_short_term()
                memory_manager.disable_short_term = True
                
                # test: Ask recall question
                print(f"   Query: {test['query']}")
                start_time = time.time()
                result = query_agent(
                    test["query"], 
                    verbose=False,
                    auto_approve=True, 
                    skip_memory=False,
                    use_custom_memory=self.use_custom_memory
                )
                latency = time.time() - start_time
                
                # Re-enable short-term memory after test
                memory_manager.disable_short_term = False
                
                # evaluate for correct memory use
                test_case = LLMTestCase(
                    input=test["query"],
                    actual_output=result["response"]
                )
                
                print(f"   Evaluating memory retrieval")
                memory_metric = MemoryRetrievalMetric(
                    expected_memory_facts=test["expected_facts"],
                    model=self.ollama_model
                )
                
                memory_score = memory_metric.measure(test_case)
                
                test_result = {
                    "test_id": test_id,
                    "fact_type": test["fact_type"],
                    "query": test["query"],
                    "expected_facts": test["expected_facts"],
                    "memory_score": round(memory_score, 3),
                    "memory_reason": memory_metric.reason,
                    "latency": round(latency, 2),
                    "timestamp": datetime.now().isoformat()
                }
                
                self.save_test_case_result(test_result)
                
                print(f"   Completed:")
                print(f"      Memory Score: {memory_score:.3f}")
                print(f"      Reason: {memory_metric.reason}")
                print(f"      Latency: {latency:.2f}s")
                
            except Exception as e:
                print(f"   Error: {e}")
                error_result = {
                    "test_id": test_id,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                self.save_test_case_result(error_result)
    
    def evaluate_fact_extraction(self, existing_results: Dict[str, List[Dict]]):
        """Evaluate fact extraction accuracy"""
        test_cases = TestDatasetGenerator.generate_fact_extraction_test_cases()
        completed_ids = self.get_completed_test_ids(existing_results, "fact_extraction")
        
        print("RUNNING FACT EXTRACTION EVALUATION")
        
        for i, test in enumerate(test_cases, 1):
            test_id = test["test_id"]
            
            # skip if already completed
            if test_id in completed_ids:
                print(f"[{i}/{len(test_cases)}] Skipping (already completed): {test_id}")
                continue
            
            print(f"\n[{i}/{len(test_cases)}] Testing: {test_id}")
            
            try:
                # run agent (fact extraction happens in step 8)
                start_time = time.time()
                result = query_agent(
                    test["input"], 
                    verbose=False,
                    auto_approve=True, 
                    skip_memory=True,
                    use_custom_memory=self.use_custom_memory
                )
                latency = time.time() - start_time
                
                # get extracted facts from result
                extracted_facts = result.get("extracted_facts", [])
                
                test_case = LLMTestCase(
                    input=test["input"],
                    actual_output=json.dumps({"extracted_facts": extracted_facts})
                )
                
                print(f"    Evaluating fact extraction")
                extraction_metric = FactExtractionAccuracyMetric(
                    expected_facts=test["expected_facts"],
                    model=self.ollama_model
                )
                
                extraction_score = extraction_metric.measure(test_case)
                
                test_result = {
                    "test_id": test_id,
                    "input": test["input"],
                    "expected_facts": test["expected_facts"],
                    "extracted_facts": extracted_facts,
                    "extraction_score": round(extraction_score, 3),
                    "extraction_reason": extraction_metric.reason,
                    "latency": round(latency, 2),
                    "timestamp": datetime.now().isoformat()
                }
                
                self.save_test_case_result(test_result)
                
                print(f"   Completed:")
                print(f"      Extraction Score: {extraction_score:.3f}")
                print(f"      Reason: {extraction_metric.reason}")
                print(f"      Latency: {latency:.2f}s")
                
            except Exception as e:
                print(f"    Error: {e}")
                error_result = {
                    "test_id": test_id,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                self.save_test_case_result(error_result)

    def print_summary(self):
        """Print evaluation summary for all test types"""

        all_results = self.load_existing_run(self.current_run_file)
        
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        
        # RAG Quality Summary
        rag_results = [r for r in all_results["rag"] if "error" not in r]
        if rag_results:
            print("\n  RAG QUALITY METRICS (n={})".format(len(rag_results)))
            avg_relevancy = sum(r["relevancy_score"] for r in rag_results) / len(rag_results)
            avg_faithfulness = sum(r["faithfulness_score"] for r in rag_results) / len(rag_results)
            avg_hallucination = sum(r["hallucination_score"] for r in rag_results) / len(rag_results)
            avg_latency = sum(r["latency"] for r in rag_results) / len(rag_results)
            avg_citations = sum(r["citation_count"] for r in rag_results) / len(rag_results)
            
            print(f"    Average Relevancy:     {avg_relevancy:.3f}")
            print(f"    Average Faithfulness:  {avg_faithfulness:.3f}")
            print(f"    Average Hallucination: {avg_hallucination:.3f}")
            print(f"    Average Latency:       {avg_latency:.2f}s")
            print(f"    Average Citations:     {avg_citations:.1f}")
        
        # Memory Retrieval Summary
        memory_results = [r for r in all_results["memory"] if "error" not in r]
        if memory_results:
            print("\n  MEMORY RETRIEVAL METRICS (n={})".format(len(memory_results)))
            avg_memory_score = sum(r["memory_score"] for r in memory_results) / len(memory_results)
            avg_latency = sum(r["latency"] for r in memory_results) / len(memory_results)
            
            print(f"    Average Memory Score:  {avg_memory_score:.3f}")
            print(f"    Average Latency:       {avg_latency:.2f}s")
            
            by_type = {}
            for r in memory_results:
                ft = r.get("fact_type", "unknown")
                if ft not in by_type:
                    by_type[ft] = []
                by_type[ft].append(r["memory_score"])
            
            print("\n    By Fact Type:")
            for fact_type, scores in by_type.items():
                avg = sum(scores) / len(scores)
                print(f"      {fact_type}: {avg:.3f} (n={len(scores)})")
        
        # Fact Extraction Summary
        fact_results = [r for r in all_results["fact_extraction"] if "error" not in r]
        if fact_results:
            print("\n  FACT EXTRACTION METRICS (n={})".format(len(fact_results)))
            avg_extraction_score = sum(r["extraction_score"] for r in fact_results) / len(fact_results)
            avg_latency = sum(r["latency"] for r in fact_results) / len(fact_results)
            
            print(f"    Average Extraction Score: {avg_extraction_score:.3f}")
            print(f"    Average Latency:          {avg_latency:.2f}s")
        
        # Overall Summary
        total_tests = sum(len(v) for v in all_results.values())
        total_errors = sum(len([r for r in v if "error" in r]) for v in all_results.values())
        
        print("\n   OVERALL SUMMARY")
        print(f"        Total Tests Run:    {total_tests}")
        print(f"        Successful:         {total_tests - total_errors}")
        print(f"        Errors:             {total_errors}")
        
        print(f"\nResults saved to: {self.current_run_file}")
        print("=" * 80 + "\n")


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI Compliance Agent Evaluations")
    parser.add_argument("--custom-memory", action="store_true", help="Use local custom memory (ChromaDB) instead of Mem0")
    
    args = parser.parse_args()
    
    runner = EvaluationRunner(use_custom_memory=args.custom_memory)
    runner.run_evaluation()