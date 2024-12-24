"""
IntegrationManager: Real-time IDE and editor integrations
"""

import logging
import asyncio
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class EditorContext:
    """Context information from the editor"""
    file_path: str
    cursor_position: Dict[str, int]
    selected_text: str
    current_line: str
    language: str
    project_files: List[str]

class IntegrationManager:
    """Manages real-time integrations with editors and IDEs"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
    
    async def handle_completion_request(self, context: EditorContext) -> Dict[str, Any]:
        """Handle code completion request from editor"""
        try:
            # Analyze context
            suggestions = await self._generate_completions(context)
            
            return {
                'suggestions': suggestions,
                'position': context.cursor_position,
                'language': context.language
            }
            
        except Exception as e:
            self.logger.error(f"Completion request failed: {str(e)}")
            return {'suggestions': [], 'error': str(e)}
    
    async def handle_real_time_analysis(self, context: EditorContext, code: str) -> Dict[str, Any]:
        """Provide real-time code analysis as user types"""
        try:
            from .code_analyzer import CodeAnalyzer
            
            analyzer = CodeAnalyzer()
            analysis = await analyzer.analyze_code(code, context.language)
            
            # Filter for real-time relevant issues
            real_time_issues = self._filter_real_time_issues(analysis.get('issues', []))
            
            return {
                'issues': real_time_issues,
                'suggestions': analysis.get('suggestions', [])[:3],  # Top 3
                'quality_score': analysis.get('quality_metrics', {})
            }
            
        except Exception as e:
            self.logger.error(f"Real-time analysis failed: {str(e)}")
            return {'issues': [], 'suggestions': []}
    
    async def _generate_completions(self, context: EditorContext) -> List[Dict[str, Any]]:
        """Generate intelligent code completions"""
        completions = []
        
        # Context-aware completions based on language
        if context.language == 'python':
            completions.extend(self._get_python_completions(context))
        elif context.language == 'javascript':
            completions.extend(self._get_javascript_completions(context))
        
        return completions
    
    def _get_python_completions(self, context: EditorContext) -> List[Dict[str, Any]]:
        """Get Python-specific completions"""
        completions = []
        current_line = context.current_line.strip()
        
        # Import suggestions
        if current_line.startswith('import ') or current_line.startswith('from '):
            completions.extend([
                {'text': 'numpy as np', 'type': 'module'},
                {'text': 'pandas as pd', 'type': 'module'},
                {'text': 'matplotlib.pyplot as plt', 'type': 'module'}
            ])
        
        # Function signatures
        if 'def ' in current_line:
            completions.extend([
                {'text': 'def __init__(self):', 'type': 'function'},
                {'text': 'def __str__(self):', 'type': 'function'}
            ])
        
        return completions
    
    def _get_javascript_completions(self, context: EditorContext) -> List[Dict[str, Any]]:
        """Get JavaScript-specific completions"""
        completions = []
        current_line = context.current_line.strip()
        
        # Console methods
        if 'console.' in current_line:
            completions.extend([
                {'text': 'log()', 'type': 'method'},
                {'text': 'error()', 'type': 'method'},
                {'text': 'warn()', 'type': 'method'}
            ])
        
        # Arrow functions
        if '=>' in current_line or 'function' in current_line:
            completions.extend([
                {'text': '(param) => { }', 'type': 'function'},
                {'text': 'async (param) => { }', 'type': 'function'}
            ])
        
        return completions
    
    def _filter_real_time_issues(self, issues: List[str]) -> List[str]:
        """Filter issues that are relevant for real-time feedback"""
        # Only show critical issues in real-time to avoid noise
        critical_keywords = ['syntax', 'error', 'security', 'performance']
        
        return [issue for issue in issues 
                if any(keyword in issue.lower() for keyword in critical_keywords)] 