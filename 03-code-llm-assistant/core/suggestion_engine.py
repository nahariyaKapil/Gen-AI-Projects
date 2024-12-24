"""
SuggestionEngine: Intelligent code suggestions and improvements
"""

import logging
from typing import Dict, List, Any

class SuggestionEngine:
    """Generate intelligent suggestions for code improvement"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def generate_suggestions(self, code: str, language: str, analysis: Dict[str, Any]) -> List[str]:
        """Generate contextual suggestions for code improvement"""
        suggestions = []
        
        try:
            # Performance suggestions
            suggestions.extend(self._get_performance_suggestions(code, language))
            
            # Style suggestions
            suggestions.extend(self._get_style_suggestions(code, language, analysis))
            
            # Best practices suggestions
            suggestions.extend(self._get_best_practices_suggestions(code, language))
            
            # Security suggestions
            suggestions.extend(self._get_security_suggestions(code, language))
            
            return suggestions[:10]  # Limit to top 10 suggestions
            
        except Exception as e:
            self.logger.error(f"Suggestion generation failed: {str(e)}")
            return ["Consider reviewing code for potential improvements"]
    
    def _get_performance_suggestions(self, code: str, language: str) -> List[str]:
        """Generate performance-related suggestions"""
        suggestions = []
        
        if language == 'python':
            if 'for i in range(len(' in code:
                suggestions.append("Consider using enumerate() instead of range(len())")
            
            if '+ str(' in code:
                suggestions.append("Consider using f-strings for string formatting")
            
            if '.append(' in code and 'for ' in code:
                suggestions.append("Consider using list comprehension for better performance")
        
        elif language == 'javascript':
            if 'document.getElementById' in code:
                suggestions.append("Consider caching DOM queries for better performance")
            
            if 'var ' in code:
                suggestions.append("Use 'const' or 'let' instead of 'var' for better performance")
        
        return suggestions
    
    def _get_style_suggestions(self, code: str, language: str, analysis: Dict[str, Any]) -> List[str]:
        """Generate style-related suggestions"""
        suggestions = []
        
        # Check line length
        if 'quality_metrics' in analysis:
            avg_line_length = analysis['quality_metrics'].get('avg_line_length', 0)
            if avg_line_length > 100:
                suggestions.append("Consider breaking long lines for better readability")
        
        # Check comment density
        if 'quality_metrics' in analysis:
            comment_density = analysis['quality_metrics'].get('comment_density', 0)
            if comment_density < 0.1:
                suggestions.append("Add more comments to explain complex logic")
        
        if language == 'python':
            if 'def ' in code and not code.count('\n\n'):
                suggestions.append("Add blank lines between function definitions (PEP 8)")
        
        return suggestions
    
    def _get_best_practices_suggestions(self, code: str, language: str) -> List[str]:
        """Generate best practices suggestions"""
        suggestions = []
        
        if language == 'python':
            if 'except:' in code:
                suggestions.append("Avoid bare except clauses; catch specific exceptions")
            
            if 'global ' in code:
                suggestions.append("Minimize use of global variables")
            
            if 'import *' in code:
                suggestions.append("Avoid wildcard imports; import specific functions")
        
        elif language == 'javascript':
            if 'eval(' in code:
                suggestions.append("Avoid using eval() for security reasons")
            
            if '== ' in code:
                suggestions.append("Use strict equality (===) instead of loose equality (==)")
        
        return suggestions
    
    def _get_security_suggestions(self, code: str, language: str) -> List[str]:
        """Generate security-related suggestions"""
        suggestions = []
        
        # Common security issues
        if 'password' in code.lower() and ('=' in code or 'input(' in code):
            suggestions.append("Avoid hardcoding passwords; use environment variables")
        
        if 'sql' in code.lower() and ('+' in code or 'format(' in code):
            suggestions.append("Use parameterized queries to prevent SQL injection")
        
        if language == 'javascript':
            if 'innerHTML' in code:
                suggestions.append("Be cautious with innerHTML; consider textContent for text")
        
        return suggestions 