"""
CodeAnalyzer: Static code analysis and quality assessment
"""

import logging
import re
from typing import Dict, List, Any
from .language_processors import get_language_processor

class CodeAnalyzer:
    """Advanced code analyzer for quality assessment and issue detection"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def analyze_code(self, code: str, language: str) -> Dict[str, Any]:
        """Perform comprehensive code analysis"""
        try:
            processor = get_language_processor(language)
            
            # Basic analysis
            analysis = {
                'language': language,
                'line_count': len(code.split('\n')),
                'character_count': len(code),
                'functions': processor.extract_functions(code),
                'syntax_validation': processor.validate_syntax(code),
                'complexity': self._calculate_complexity(code),
                'issues': self._detect_issues(code, language),
                'quality_metrics': self._calculate_quality_metrics(code, language)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Code analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_complexity(self, code: str) -> str:
        """Calculate code complexity"""
        lines = [line.strip() for line in code.split('\n') if line.strip()]
        complexity_indicators = ['if', 'elif', 'else', 'for', 'while', 'try', 'except', 'case', 'switch']
        
        complexity_count = sum(1 for line in lines 
                              for indicator in complexity_indicators 
                              if indicator in line.lower())
        
        if complexity_count < 5:
            return 'low'
        elif complexity_count < 15:
            return 'medium'
        else:
            return 'high'
    
    def _detect_issues(self, code: str, language: str) -> List[str]:
        """Detect common code issues"""
        issues = []
        
        # Common issues across languages
        if len(code.split('\n')) > 100:
            issues.append("Code is very long, consider breaking into smaller functions")
        
        if language == 'python':
            issues.extend(self._detect_python_issues(code))
        elif language == 'javascript':
            issues.extend(self._detect_javascript_issues(code))
        
        return issues
    
    def _detect_python_issues(self, code: str) -> List[str]:
        """Detect Python-specific issues"""
        issues = []
        
        # Long lines
        for i, line in enumerate(code.split('\n'), 1):
            if len(line) > 79:
                issues.append(f"Line {i} exceeds 79 characters (PEP 8)")
        
        # Missing docstrings
        if 'def ' in code and '"""' not in code and "'''" not in code:
            issues.append("Functions should have docstrings")
        
        # Unused imports
        import_lines = [line for line in code.split('\n') if line.strip().startswith('import')]
        for import_line in import_lines:
            module = import_line.split()[-1]
            if module not in code.replace(import_line, ''):
                issues.append(f"Unused import: {module}")
        
        return issues
    
    def _detect_javascript_issues(self, code: str) -> List[str]:
        """Detect JavaScript-specific issues"""
        issues = []
        
        # Missing semicolons
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if (stripped and 
                not stripped.endswith((';', '{', '}')) and
                not stripped.startswith(('//', '/*'))):
                issues.append(f"Line {i} might be missing semicolon")
        
        # Use of var instead of let/const
        if 'var ' in code:
            issues.append("Consider using 'let' or 'const' instead of 'var'")
        
        return issues
    
    def _calculate_quality_metrics(self, code: str, language: str) -> Dict[str, Any]:
        """Calculate various quality metrics"""
        lines = [line for line in code.split('\n') if line.strip()]
        
        # Comment density
        comment_lines = [line for line in lines if line.strip().startswith(('#', '//', '/*'))]
        comment_density = len(comment_lines) / max(len(lines), 1)
        
        # Average line length
        avg_line_length = sum(len(line) for line in lines) / max(len(lines), 1)
        
        return {
            'comment_density': comment_density,
            'avg_line_length': avg_line_length,
            'function_count': len(get_language_processor(language).extract_functions(code))
        } 