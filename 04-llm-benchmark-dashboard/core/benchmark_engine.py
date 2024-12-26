"""
BenchmarkEngine: Comprehensive LLM performance testing and evaluation
"""

import logging
import time
import asyncio
import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import openai
from datetime import datetime

@dataclass
class BenchmarkTest:
    """Individual benchmark test configuration"""
    name: str
    category: str
    prompt: str
    expected_output_type: str
    evaluation_criteria: List[str]
    timeout: float = 30.0
    max_tokens: int = 500

@dataclass
class BenchmarkResult:
    """Results from a benchmark test"""
    test_name: str
    model_name: str
    response_time: float
    token_count: int
    cost: float
    quality_score: float
    error_rate: float
    throughput: float
    memory_usage: float
    response: str
    timestamp: datetime = field(default_factory=datetime.now)

class BenchmarkCategory(Enum):
    REASONING = "reasoning"
    CODE_GENERATION = "code_generation"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    CREATIVE_WRITING = "creative_writing"
    QUESTION_ANSWERING = "question_answering"
    CLASSIFICATION = "classification"

class BenchmarkEngine:
    """
    Advanced LLM benchmarking engine for comprehensive performance evaluation
    
    Features:
    - Multi-model comparison
    - Various test categories
    - Performance optimization
    - Cost analysis
    - Real-time monitoring
    - Quality assessment
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize OpenAI client with proper error handling
        try:
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                # Use a mock client for demo purposes
                self.openai_client = None
                self.logger.warning("OpenAI API key not found. Using mock mode.")
            else:
                self.openai_client = openai.OpenAI(api_key=api_key)
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            self.openai_client = None
        
        # Test suites
        self.test_suites = self._initialize_test_suites()
        
        # Results storage
        self.benchmark_results: List[BenchmarkResult] = []
        self.model_performance: Dict[str, Dict[str, float]] = {}
        
        # Model configurations
        self.models_to_test = [
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4-turbo-preview"
        ]
        
        self.logger.info("BenchmarkEngine initialized successfully")

    async def run_comprehensive_benchmark(self, models: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run comprehensive benchmark across all test categories"""
        models = models or self.models_to_test
        
        try:
            self.logger.info(f"Starting comprehensive benchmark for {len(models)} models")
            
            overall_results = {
                'summary': {},
                'detailed_results': [],
                'performance_comparison': {},
                'cost_analysis': {},
                'recommendations': []
            }
            
            # Run tests for each model
            for model in models:
                self.logger.info(f"Testing model: {model}")
                model_results = await self._test_model_comprehensive(model)
                overall_results['detailed_results'].extend(model_results)
            
            # Analyze results
            overall_results['summary'] = self._generate_summary(overall_results['detailed_results'])
            overall_results['performance_comparison'] = self._compare_models(overall_results['detailed_results'])
            overall_results['cost_analysis'] = self._analyze_costs(overall_results['detailed_results'])
            overall_results['recommendations'] = self._generate_recommendations(overall_results['detailed_results'])
            
            self.logger.info("Comprehensive benchmark completed successfully")
            return overall_results
            
        except Exception as e:
            self.logger.error(f"Benchmark failed: {str(e)}")
            raise

    async def run_category_benchmark(self, category: BenchmarkCategory, models: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run benchmark for specific category"""
        models = models or self.models_to_test
        
        try:
            category_tests = [test for test in self.test_suites if test.category == category.value]
            results = []
            
            for model in models:
                for test in category_tests:
                    result = await self._run_single_test(model, test)
                    results.append(result)
            
            return {
                'category': category.value,
                'results': results,
                'analysis': self._analyze_category_results(results)
            }
            
        except Exception as e:
            self.logger.error(f"Category benchmark failed: {str(e)}")
            raise

    async def run_stress_test(self, model: str, concurrent_requests: int = 10, duration: int = 60) -> Dict[str, Any]:
        """Run stress test with concurrent requests"""
        try:
            self.logger.info(f"Starting stress test for {model} with {concurrent_requests} concurrent requests")
            
            start_time = time.time()
            completed_requests = 0
            failed_requests = 0
            response_times = []
            
            # Simple stress test prompt
            test_prompt = "Explain quantum computing in simple terms."
            
            async def stress_request():
                nonlocal completed_requests, failed_requests
                try:
                    request_start = time.time()
                    
                    response = self.openai_client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": test_prompt}],
                        max_tokens=100
                    )
                    
                    request_time = time.time() - request_start
                    response_times.append(request_time)
                    completed_requests += 1
                    
                except Exception as e:
                    failed_requests += 1
                    self.logger.error(f"Stress test request failed: {str(e)}")
            
            # Run concurrent requests for specified duration
            tasks = []
            while time.time() - start_time < duration:
                if len(tasks) < concurrent_requests:
                    task = asyncio.create_task(stress_request())
                    tasks.append(task)
                
                # Clean up completed tasks
                tasks = [task for task in tasks if not task.done()]
                
                await asyncio.sleep(0.1)  # Small delay
            
            # Wait for remaining tasks
            await asyncio.gather(*tasks, return_exceptions=True)
            
            total_time = time.time() - start_time
            
            return {
                'model': model,
                'duration': total_time,
                'completed_requests': completed_requests,
                'failed_requests': failed_requests,
                'success_rate': completed_requests / (completed_requests + failed_requests) if (completed_requests + failed_requests) > 0 else 0,
                'avg_response_time': np.mean(response_times) if response_times else 0,
                'median_response_time': np.median(response_times) if response_times else 0,
                'p95_response_time': np.percentile(response_times, 95) if response_times else 0,
                'throughput': completed_requests / total_time if total_time > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Stress test failed: {str(e)}")
            raise

    async def _test_model_comprehensive(self, model: str) -> List[BenchmarkResult]:
        """Test a model across all test suites"""
        results = []
        
        for test in self.test_suites:
            try:
                result = await self._run_single_test(model, test)
                results.append(result)
                
                # Add delay to avoid rate limiting
                await asyncio.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"Test {test.name} failed for {model}: {str(e)}")
                
                # Create error result
                error_result = BenchmarkResult(
                    test_name=test.name,
                    model_name=model,
                    response_time=0.0,
                    token_count=0,
                    cost=0.0,
                    quality_score=0.0,
                    error_rate=1.0,
                    throughput=0.0,
                    memory_usage=0.0,
                    response=f"Error: {str(e)}"
                )
                results.append(error_result)
        
        return results

    async def _run_single_test(self, model: str, test: BenchmarkTest) -> BenchmarkResult:
        """Run a single benchmark test"""
        start_time = time.time()
        
        try:
            # Check if OpenAI client is available
            if self.openai_client is None:
                # Mock response for demo purposes
                response_time = 0.5 + np.random.uniform(0, 1)  # Random response time
                response_text = f"[DEMO MODE] Mock response for {test.name} using {model}"
                token_count = 100 + np.random.randint(0, 200)  # Random token count
            else:
                # Make API call
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": test.prompt}],
                    max_tokens=test.max_tokens,
                    temperature=0.7
                )
                
                response_time = time.time() - start_time
                
                # Extract response details
                response_text = response.choices[0].message.content
                token_count = response.usage.total_tokens if response.usage else 0
            
            # Calculate cost (simplified pricing)
            cost = self._calculate_cost(model, token_count)
            
            # Evaluate quality
            quality_score = self._evaluate_response_quality(response_text, test)
            
            result = BenchmarkResult(
                test_name=test.name,
                model_name=model,
                response_time=response_time,
                token_count=token_count,
                cost=cost,
                quality_score=quality_score,
                error_rate=0.0,
                throughput=token_count / response_time if response_time > 0 else 0,
                memory_usage=0.0,  # Would need system monitoring
                response=response_text
            )
            
            self.benchmark_results.append(result)
            return result
            
        except Exception as e:
            response_time = time.time() - start_time
            self.logger.error(f"Single test failed: {str(e)}")
            
            return BenchmarkResult(
                test_name=test.name,
                model_name=model,
                response_time=response_time,
                token_count=0,
                cost=0.0,
                quality_score=0.0,
                error_rate=1.0,
                throughput=0.0,
                memory_usage=0.0,
                response=f"Error: {str(e)}"
            )

    def _initialize_test_suites(self) -> List[BenchmarkTest]:
        """Initialize comprehensive test suites"""
        tests = []
        
        # Reasoning tests
        tests.extend([
            BenchmarkTest(
                name="logical_reasoning",
                category="reasoning",
                prompt="If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
                expected_output_type="logical_analysis",
                evaluation_criteria=["logical_correctness", "explanation_clarity"]
            ),
            BenchmarkTest(
                name="mathematical_reasoning",
                category="reasoning", 
                prompt="A train travels 120 km in 2 hours. If it maintains the same speed, how far will it travel in 5 hours?",
                expected_output_type="calculation",
                evaluation_criteria=["mathematical_accuracy", "step_by_step_explanation"]
            )
        ])
        
        # Code generation tests
        tests.extend([
            BenchmarkTest(
                name="python_function",
                category="code_generation",
                prompt="Write a Python function that finds the factorial of a number using recursion.",
                expected_output_type="code",
                evaluation_criteria=["syntax_correctness", "logic_correctness", "efficiency"]
            ),
            BenchmarkTest(
                name="javascript_sorting",
                category="code_generation",
                prompt="Create a JavaScript function that sorts an array of objects by a specified property.",
                expected_output_type="code",
                evaluation_criteria=["syntax_correctness", "functionality", "best_practices"]
            )
        ])
        
        # Summarization tests
        tests.extend([
            BenchmarkTest(
                name="article_summary",
                category="summarization",
                prompt="Summarize this text in 3 sentences: 'Artificial intelligence has transformed many industries over the past decade. From healthcare to finance, AI systems are now capable of performing complex tasks that were once exclusive to human experts. Machine learning algorithms can analyze vast amounts of data to identify patterns and make predictions. However, the rapid advancement of AI also raises important questions about job displacement, privacy, and ethical considerations that society must address.'",
                expected_output_type="summary",
                evaluation_criteria=["conciseness", "accuracy", "key_points_coverage"]
            )
        ])
        
        # Translation tests
        tests.extend([
            BenchmarkTest(
                name="english_to_spanish",
                category="translation",
                prompt="Translate this English text to Spanish: 'The weather is beautiful today, and I plan to go for a walk in the park.'",
                expected_output_type="translation",
                evaluation_criteria=["accuracy", "fluency", "cultural_appropriateness"]
            )
        ])
        
        return tests

    def _calculate_cost(self, model: str, token_count: int) -> float:
        """Calculate cost based on model and token usage"""
        # Simplified pricing (per 1000 tokens)
        pricing = {
            "gpt-3.5-turbo": 0.0015,
            "gpt-4": 0.03,
            "gpt-4-turbo-preview": 0.01
        }
        
        price_per_1k = pricing.get(model, 0.002)
        return (token_count / 1000) * price_per_1k

    def _evaluate_response_quality(self, response: str, test: BenchmarkTest) -> float:
        """Evaluate response quality based on test criteria"""
        # Simplified quality scoring
        base_score = 0.7
        
        # Length appropriateness
        if len(response) < 10:
            base_score -= 0.3
        elif len(response) > 1000:
            base_score -= 0.1
        
        # Category-specific evaluation
        if test.category == "code_generation":
            if "def " in response or "function " in response:
                base_score += 0.2
            if "```" in response:
                base_score += 0.1
        
        elif test.category == "reasoning":
            if "because" in response.lower() or "therefore" in response.lower():
                base_score += 0.2
        
        return min(1.0, max(0.0, base_score))

    def _generate_summary(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate summary statistics from benchmark results"""
        if not results:
            return {}
        
        # Group by model
        by_model = {}
        for result in results:
            if result.model_name not in by_model:
                by_model[result.model_name] = []
            by_model[result.model_name].append(result)
        
        summary = {}
        for model, model_results in by_model.items():
            summary[model] = {
                'avg_response_time': np.mean([r.response_time for r in model_results]),
                'avg_quality_score': np.mean([r.quality_score for r in model_results]),
                'total_cost': sum([r.cost for r in model_results]),
                'avg_throughput': np.mean([r.throughput for r in model_results]),
                'success_rate': 1 - np.mean([r.error_rate for r in model_results]),
                'total_tests': len(model_results)
            }
        
        return summary

    def _compare_models(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Compare performance across models"""
        comparison = {}
        
        # Group by test
        by_test = {}
        for result in results:
            if result.test_name not in by_test:
                by_test[result.test_name] = []
            by_test[result.test_name].append(result)
        
        # Find best performer for each test
        for test_name, test_results in by_test.items():
            best_quality = max(test_results, key=lambda x: x.quality_score)
            fastest = min(test_results, key=lambda x: x.response_time)
            cheapest = min(test_results, key=lambda x: x.cost)
            
            comparison[test_name] = {
                'best_quality': {'model': best_quality.model_name, 'score': best_quality.quality_score},
                'fastest': {'model': fastest.model_name, 'time': fastest.response_time},
                'cheapest': {'model': cheapest.model_name, 'cost': cheapest.cost}
            }
        
        return comparison

    def _analyze_costs(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Analyze cost patterns across models and tests"""
        cost_analysis = {}
        
        # Total cost by model
        by_model = {}
        for result in results:
            if result.model_name not in by_model:
                by_model[result.model_name] = []
            by_model[result.model_name].append(result.cost)
        
        for model, costs in by_model.items():
            cost_analysis[model] = {
                'total_cost': sum(costs),
                'avg_cost_per_request': np.mean(costs),
                'cost_efficiency': np.mean([r.quality_score / max(r.cost, 0.0001) for r in results if r.model_name == model])
            }
        
        return cost_analysis

    def _generate_recommendations(self, results: List[BenchmarkResult]) -> List[str]:
        """Generate optimization recommendations based on results"""
        recommendations = []
        
        # Analyze results for recommendations
        model_performance = self._generate_summary(results)
        
        if model_performance:
            # Find best overall performer
            best_model = max(model_performance.keys(), 
                           key=lambda m: model_performance[m]['avg_quality_score'])
            recommendations.append(f"Best overall quality: {best_model}")
            
            # Find most cost-effective
            cheapest_model = min(model_performance.keys(),
                               key=lambda m: model_performance[m]['total_cost'])
            recommendations.append(f"Most cost-effective: {cheapest_model}")
            
            # Performance recommendations
            for model, stats in model_performance.items():
                if stats['avg_response_time'] > 5.0:
                    recommendations.append(f"Consider optimization for {model} - slow response times")
                
                if stats['success_rate'] < 0.9:
                    recommendations.append(f"Reliability concerns with {model}")
        
        return recommendations

    def _analyze_category_results(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Analyze results for a specific category"""
        if not results:
            return {}
        
        return {
            'avg_quality': np.mean([r.quality_score for r in results]),
            'avg_response_time': np.mean([r.response_time for r in results]),
            'total_cost': sum([r.cost for r in results]),
            'success_rate': 1 - np.mean([r.error_rate for r in results]),
            'model_comparison': {
                model: {
                    'quality': np.mean([r.quality_score for r in results if r.model_name == model]),
                    'speed': np.mean([r.response_time for r in results if r.model_name == model])
                }
                for model in set(r.model_name for r in results)
            }
        }

    def get_historical_performance(self, model: str, days: int = 30) -> Dict[str, Any]:
        """Get historical performance data for a model"""
        # In production, this would query a database
        recent_results = [r for r in self.benchmark_results 
                         if r.model_name == model and 
                         (datetime.now() - r.timestamp).days <= days]
        
        if not recent_results:
            return {}
        
        return {
            'model': model,
            'period_days': days,
            'total_tests': len(recent_results),
            'avg_quality': np.mean([r.quality_score for r in recent_results]),
            'avg_response_time': np.mean([r.response_time for r in recent_results]),
            'total_cost': sum([r.cost for r in recent_results]),
            'quality_trend': self._calculate_trend([r.quality_score for r in recent_results]),
            'performance_trend': self._calculate_trend([1/r.response_time for r in recent_results])
        }

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 2:
            return "insufficient_data"
        
        first_half = np.mean(values[:len(values)//2])
        second_half = np.mean(values[len(values)//2:])
        
        if second_half > first_half * 1.05:
            return "improving"
        elif second_half < first_half * 0.95:
            return "declining"
        else:
            return "stable" 