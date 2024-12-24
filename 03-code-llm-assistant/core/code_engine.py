"""
CodeEngine: Main orchestrator for code generation and assistance
"""

import logging
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import openai
import ast
import subprocess
import tempfile
import os

from .language_processors import LanguageProcessor, get_language_processor
from .code_analyzer import CodeAnalyzer
from .suggestion_engine import SuggestionEngine
try:
    from ..shared_infrastructure.monitoring import MetricsCollector
except ImportError:
    # Fallback - create a simple mock class
    class MetricsCollector:
        def __init__(self):
            pass
        def increment_counter(self, name):
            pass
        def record_timing(self, name, value):
            pass
try:
    from ..shared_infrastructure.config import get_config
except ImportError:
    # Fallback - create a simple mock config function
    def get_config():
        return {
            'openai_api_key': 'demo-key',
            'code_model': 'gpt-4'
        }

class CodeQuality(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

@dataclass
class CodeGenerationRequest:
    """Request for code generation"""
    task_description: str
    language: str
    context: Optional[str] = None
    requirements: List[str] = None
    style_preferences: Dict[str, Any] = None
    target_framework: Optional[str] = None

@dataclass
class CodeGenerationResponse:
    """Response from code generation"""
    code: str
    language: str
    quality_score: float
    suggestions: List[str]
    execution_time: float
    analysis_results: Dict[str, Any]
    test_results: Optional[Dict[str, Any]] = None

class CodeEngine:
    """
    Advanced code generation and assistance engine
    
    Features:
    - Multi-language code generation
    - Real-time code analysis
    - Quality assessment
    - Intelligent suggestions
    - Test generation and execution
    - Performance optimization
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.metrics = MetricsCollector()
        
        # Initialize components
        self.code_analyzer = CodeAnalyzer()
        self.suggestion_engine = SuggestionEngine()
        
        # LLM client
        self.openai_client = openai.OpenAI(
            api_key=self.config.get('openai_api_key')
        )
        
        # Supported languages
        self.supported_languages = [
            'python', 'javascript', 'typescript', 'java', 
            'cpp', 'go', 'rust', 'csharp'
        ]
        
        # Performance tracking
        self.generation_history: List[Dict[str, Any]] = []
        self.language_stats = {lang: {'count': 0, 'avg_quality': 0.0} 
                             for lang in self.supported_languages}
        
        self.logger.info("CodeEngine initialized successfully")

    async def generate_code(self, request: CodeGenerationRequest) -> CodeGenerationResponse:
        """Generate code based on request"""
        start_time = time.time()
        
        try:
            self.logger.info(f"Generating {request.language} code: {request.task_description[:50]}...")
            
            # Validate request
            if not self._validate_request(request):
                raise ValueError("Invalid code generation request")
            
            # Generate code using LLM
            generated_code = await self._generate_with_llm(request)
            
            # Analyze generated code
            analysis_results = await self.code_analyzer.analyze_code(
                generated_code, request.language
            )
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(analysis_results)
            
            # Generate suggestions
            suggestions = await self.suggestion_engine.generate_suggestions(
                generated_code, request.language, analysis_results
            )
            
            # Run tests if applicable
            test_results = None
            if request.language in ['python', 'javascript']:
                test_results = await self._run_basic_tests(generated_code, request.language)
            
            execution_time = time.time() - start_time
            
            # Track metrics
            self._update_metrics(request.language, quality_score, execution_time)
            
            response = CodeGenerationResponse(
                code=generated_code,
                language=request.language,
                quality_score=quality_score,
                suggestions=suggestions,
                execution_time=execution_time,
                analysis_results=analysis_results,
                test_results=test_results
            )
            
            # Store in history
            self.generation_history.append({
                'task': request.task_description,
                'language': request.language,
                'quality_score': quality_score,
                'execution_time': execution_time,
                'timestamp': time.time()
            })
            
            self.logger.info(f"Code generation completed in {execution_time:.2f}s")
            return response
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Code generation failed: {str(e)}")
            self.metrics.increment_counter('code_generation_failures')
            raise

    async def analyze_existing_code(self, code: str, language: str) -> Dict[str, Any]:
        """Analyze existing code for quality and suggestions"""
        try:
            analysis_results = await self.code_analyzer.analyze_code(code, language)
            suggestions = await self.suggestion_engine.generate_suggestions(
                code, language, analysis_results
            )
            
            return {
                'analysis': analysis_results,
                'suggestions': suggestions,
                'quality_score': self._calculate_quality_score(analysis_results)
            }
            
        except Exception as e:
            self.logger.error(f"Code analysis failed: {str(e)}")
            raise

    async def optimize_code(self, code: str, language: str) -> str:
        """Optimize existing code for performance and readability"""
        try:
            # Analyze current code
            analysis = await self.code_analyzer.analyze_code(code, language)
            
            # Generate optimization prompt
            optimization_prompt = self._create_optimization_prompt(code, language, analysis)
            
            # Get optimized code from LLM
            optimized_code = await self._call_llm_for_optimization(optimization_prompt, language)
            
            return optimized_code
            
        except Exception as e:
            self.logger.error(f"Code optimization failed: {str(e)}")
            raise

    async def generate_tests(self, code: str, language: str) -> str:
        """Generate comprehensive tests for given code"""
        try:
            test_prompt = self._create_test_generation_prompt(code, language)
            tests = await self._call_llm_for_tests(test_prompt, language)
            
            return tests
            
        except Exception as e:
            self.logger.error(f"Test generation failed: {str(e)}")
            raise

    async def _generate_with_llm(self, request: CodeGenerationRequest) -> str:
        """Generate code using LLM"""
        prompt = self._create_generation_prompt(request)
        
        response = self.openai_client.chat.completions.create(
            model=self.config.get('code_model', 'gpt-4'),
            messages=[
                {
                    "role": "system", 
                    "content": f"You are an expert {request.language} developer. Generate clean, efficient, well-documented code."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.3
        )
        
        generated_code = response.choices[0].message.content
        
        # Extract code from response (remove markdown formatting)
        if '```' in generated_code:
            code_blocks = generated_code.split('```')
            for block in code_blocks:
                if request.language.lower() in block.lower() or block.strip().startswith(('def ', 'function ', 'class ')):
                    return block.strip()
            # If no language-specific block found, take the largest code block
            code_blocks = [block.strip() for block in code_blocks if block.strip()]
            if code_blocks:
                return max(code_blocks, key=len)
        
        return generated_code.strip()

    def _create_generation_prompt(self, request: CodeGenerationRequest) -> str:
        """Create prompt for code generation"""
        prompt = f"Generate {request.language} code for the following task:\n\n"
        prompt += f"Task: {request.task_description}\n\n"
        
        if request.context:
            prompt += f"Context: {request.context}\n\n"
        
        if request.requirements:
            prompt += "Requirements:\n"
            for req in request.requirements:
                prompt += f"- {req}\n"
            prompt += "\n"
        
        if request.target_framework:
            prompt += f"Target Framework: {request.target_framework}\n\n"
        
        prompt += "Please provide:\n"
        prompt += "1. Clean, readable code with proper comments\n"
        prompt += "2. Error handling where appropriate\n"
        prompt += "3. Efficient algorithms and data structures\n"
        prompt += "4. Follow best practices for the language\n"
        
        return prompt

    def _create_optimization_prompt(self, code: str, language: str, analysis: Dict) -> str:
        """Create prompt for code optimization"""
        prompt = f"Optimize the following {language} code for better performance and readability:\n\n"
        prompt += f"```{language}\n{code}\n```\n\n"
        
        if 'issues' in analysis and analysis['issues']:
            prompt += "Identified issues to fix:\n"
            for issue in analysis['issues'][:5]:  # Top 5 issues
                prompt += f"- {issue}\n"
            prompt += "\n"
        
        prompt += "Please provide optimized code that:\n"
        prompt += "1. Fixes any performance issues\n"
        prompt += "2. Improves readability and maintainability\n"
        prompt += "3. Follows best practices\n"
        prompt += "4. Maintains the same functionality\n"
        
        return prompt

    def _create_test_generation_prompt(self, code: str, language: str) -> str:
        """Create prompt for test generation"""
        prompt = f"Generate comprehensive tests for the following {language} code:\n\n"
        prompt += f"```{language}\n{code}\n```\n\n"
        prompt += "Please provide:\n"
        prompt += "1. Unit tests covering all functions/methods\n"
        prompt += "2. Edge cases and error conditions\n"
        prompt += "3. Input validation tests\n"
        prompt += "4. Performance tests if applicable\n"
        
        if language == 'python':
            prompt += "5. Use pytest framework\n"
        elif language == 'javascript':
            prompt += "5. Use Jest framework\n"
        
        return prompt

    async def _call_llm_for_optimization(self, prompt: str, language: str) -> str:
        """Call LLM for code optimization"""
        response = self.openai_client.chat.completions.create(
            model=self.config.get('code_model', 'gpt-4'),
            messages=[
                {
                    "role": "system",
                    "content": f"You are an expert {language} developer specializing in code optimization."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.2
        )
        
        return self._extract_code_from_response(response.choices[0].message.content, language)

    async def _call_llm_for_tests(self, prompt: str, language: str) -> str:
        """Call LLM for test generation"""
        response = self.openai_client.chat.completions.create(
            model=self.config.get('code_model', 'gpt-4'),
            messages=[
                {
                    "role": "system",
                    "content": f"You are an expert {language} test engineer. Generate comprehensive, well-structured tests."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.3
        )
        
        return self._extract_code_from_response(response.choices[0].message.content, language)

    def _extract_code_from_response(self, response: str, language: str) -> str:
        """Extract code from LLM response"""
        if '```' in response:
            code_blocks = response.split('```')
            for i, block in enumerate(code_blocks):
                if language.lower() in block.lower() or (i > 0 and block.strip()):
                    return block.strip()
        return response.strip()

    async def _run_basic_tests(self, code: str, language: str) -> Dict[str, Any]:
        """Run basic tests on generated code"""
        try:
            if language == 'python':
                return await self._test_python_code(code)
            elif language == 'javascript':
                return await self._test_javascript_code(code)
            else:
                return {'status': 'skipped', 'reason': 'Language not supported for testing'}
                
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

    async def _test_python_code(self, code: str) -> Dict[str, Any]:
        """Test Python code for syntax and basic execution"""
        try:
            # Syntax check
            ast.parse(code)
            
            # Create temporary file and try to import
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                # Try to execute the code in a subprocess
                result = subprocess.run(
                    ['python', '-c', f'exec(open("{temp_file}").read())'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                return {
                    'status': 'success' if result.returncode == 0 else 'execution_error',
                    'syntax_valid': True,
                    'execution_output': result.stdout,
                    'execution_errors': result.stderr
                }
                
            finally:
                os.unlink(temp_file)
                
        except SyntaxError as e:
            return {
                'status': 'syntax_error',
                'syntax_valid': False,
                'error': str(e)
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

    async def _test_javascript_code(self, code: str) -> Dict[str, Any]:
        """Test JavaScript code for syntax"""
        try:
            # Basic syntax check using Node.js
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                result = subprocess.run(
                    ['node', '--check', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                return {
                    'status': 'success' if result.returncode == 0 else 'syntax_error',
                    'syntax_valid': result.returncode == 0,
                    'errors': result.stderr if result.stderr else None
                }
                
            finally:
                os.unlink(temp_file)
                
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

    def _validate_request(self, request: CodeGenerationRequest) -> bool:
        """Validate code generation request"""
        if not request.task_description or not request.task_description.strip():
            return False
        
        if request.language not in self.supported_languages:
            return False
        
        return True

    def _calculate_quality_score(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate code quality score based on analysis"""
        base_score = 0.8
        
        # Deduct points for issues
        if 'issues' in analysis_results:
            issue_count = len(analysis_results['issues'])
            base_score -= min(0.4, issue_count * 0.05)
        
        # Add points for good practices
        if 'complexity' in analysis_results:
            if analysis_results['complexity'] == 'low':
                base_score += 0.1
            elif analysis_results['complexity'] == 'high':
                base_score -= 0.1
        
        return max(0.0, min(1.0, base_score))

    def _update_metrics(self, language: str, quality_score: float, execution_time: float):
        """Update performance metrics"""
        self.metrics.increment_counter('code_generations_total')
        self.metrics.increment_counter(f'code_generations_{language}')
        self.metrics.record_gauge('code_quality_score', quality_score)
        self.metrics.record_gauge('code_generation_time', execution_time)
        
        # Update language stats
        if language in self.language_stats:
            stats = self.language_stats[language]
            stats['count'] += 1
            stats['avg_quality'] = (stats['avg_quality'] * (stats['count'] - 1) + quality_score) / stats['count']

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        if not self.generation_history:
            return {
                'total_generations': 0,
                'avg_quality_score': 0.0,
                'avg_execution_time': 0.0,
                'language_breakdown': self.language_stats
            }
        
        total_generations = len(self.generation_history)
        avg_quality = sum(h['quality_score'] for h in self.generation_history) / total_generations
        avg_time = sum(h['execution_time'] for h in self.generation_history) / total_generations
        
        return {
            'total_generations': total_generations,
            'avg_quality_score': avg_quality,
            'avg_execution_time': avg_time,
            'language_breakdown': self.language_stats,
            'recent_generations': self.generation_history[-10:]  # Last 10
        } 