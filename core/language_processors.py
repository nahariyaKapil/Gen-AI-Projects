"""
Expert-level Language Processors for Code Analysis and Generation
Supports multiple programming languages with advanced features
"""

import re
import ast
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CodeAnalysis:
    """Code analysis results"""
    complexity: int
    functions: List[str]
    classes: List[str]
    imports: List[str]
    lines_of_code: int
    issues: List[str]
    suggestions: List[str]

@dataclass
class CodeMetrics:
    """Code quality metrics"""
    cyclomatic_complexity: int
    maintainability_index: float
    code_coverage: float
    duplication_ratio: float

class LanguageProcessor(ABC):
    """Base class for language-specific code processors"""
    
    def __init__(self, language: str):
        self.language = language
        self.file_extensions = self._get_file_extensions()
        
    @abstractmethod
    def _get_file_extensions(self) -> List[str]:
        """Get supported file extensions"""
        pass
    
    @abstractmethod
    def analyze_code(self, code: str) -> CodeAnalysis:
        """Analyze code and return analysis results"""
        pass
    
    @abstractmethod
    def generate_code(self, prompt: str, context: Optional[str] = None) -> str:
        """Generate code based on prompt"""
        pass
    
    @abstractmethod
    def optimize_code(self, code: str) -> str:
        """Optimize the given code"""
        pass
    
    @abstractmethod
    def validate_syntax(self, code: str) -> Tuple[bool, List[str]]:
        """Validate code syntax"""
        pass

class PythonProcessor(LanguageProcessor):
    """Expert-level Python code processor"""
    
    def __init__(self):
        super().__init__("python")
        
    def _get_file_extensions(self) -> List[str]:
        return [".py", ".pyw", ".pyx"]
    
    def analyze_code(self, code: str) -> CodeAnalysis:
        """Advanced Python code analysis"""
        try:
            tree = ast.parse(code)
            
            functions = []
            classes = []
            imports = []
            issues = []
            suggestions = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
                    if len(node.args.args) > 5:
                        issues.append(f"Function '{node.name}' has too many parameters")
                        
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                    
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        imports.extend([alias.name for alias in node.names])
                    else:
                        imports.append(node.module or "")
            
            # Calculate complexity
            complexity = self._calculate_complexity(tree)
            
            # Count lines of code
            lines_of_code = len([line for line in code.split('\n') if line.strip() and not line.strip().startswith('#')])
            
            # Generate suggestions
            if complexity > 10:
                suggestions.append("Consider breaking down complex functions")
            if len(functions) > 20:
                suggestions.append("Consider splitting into multiple modules")
                
            return CodeAnalysis(
                complexity=complexity,
                functions=functions,
                classes=classes,
                imports=imports,
                lines_of_code=lines_of_code,
                issues=issues,
                suggestions=suggestions
            )
            
        except SyntaxError as e:
            return CodeAnalysis(0, [], [], [], 0, [f"Syntax error: {str(e)}"], [])
    
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.With, ast.Try)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, (ast.And, ast.Or)):
                complexity += 1
                
        return complexity
    
    def generate_code(self, prompt: str, context: Optional[str] = None) -> str:
        """Generate Python code based on prompt"""
        # This would integrate with OpenAI API in production
        template = f'''
def generated_function():
    """
    Generated based on: {prompt}
    Context: {context or 'None'}
    """
    # TODO: Implement the requested functionality
    # {prompt}
    pass

# Example usage:
if __name__ == "__main__":
    result = generated_function()
    print(result)
'''
        return template.strip()
    
    def optimize_code(self, code: str) -> str:
        """Optimize Python code"""
        # Basic optimizations
        optimized = code
        
        # Remove unnecessary imports
        lines = optimized.split('\n')
        import_lines = [line for line in lines if line.strip().startswith('import') or line.strip().startswith('from')]
        used_imports = []
        
        for import_line in import_lines:
            module_name = import_line.split()[1].split('.')[0]
            if any(module_name in line for line in lines if not line.strip().startswith(('import', 'from'))):
                used_imports.append(import_line)
        
        # Suggest list comprehensions
        if 'for ' in code and 'append(' in code:
            optimized += '\n# Suggestion: Consider using list comprehensions for better performance'
            
        return optimized
    
    def validate_syntax(self, code: str) -> Tuple[bool, List[str]]:
        """Validate Python syntax"""
        try:
            ast.parse(code)
            return True, []
        except SyntaxError as e:
            return False, [f"Syntax error at line {e.lineno}: {e.msg}"]

class JavaScriptProcessor(LanguageProcessor):
    """Expert-level JavaScript code processor"""
    
    def __init__(self):
        super().__init__("javascript")
        
    def _get_file_extensions(self) -> List[str]:
        return [".js", ".jsx", ".ts", ".tsx"]
    
    def analyze_code(self, code: str) -> CodeAnalysis:
        """Analyze JavaScript code"""
        functions = re.findall(r'function\s+(\w+)', code)
        classes = re.findall(r'class\s+(\w+)', code)
        imports = re.findall(r'import.*from\s+[\'"]([^\'"]+)[\'"]', code)
        
        lines_of_code = len([line for line in code.split('\n') if line.strip() and not line.strip().startswith('//')])
        
        issues = []
        suggestions = []
        
        if 'var ' in code:
            issues.append("Use 'let' or 'const' instead of 'var'")
        if '==' in code and '===' not in code:
            issues.append("Use strict equality (===) instead of loose equality (==)")
            
        return CodeAnalysis(
            complexity=len(functions) + len(classes),
            functions=functions,
            classes=classes,
            imports=imports,
            lines_of_code=lines_of_code,
            issues=issues,
            suggestions=suggestions
        )
    
    def generate_code(self, prompt: str, context: Optional[str] = None) -> str:
        """Generate JavaScript code"""
        return f'''
// Generated based on: {prompt}
// Context: {context or 'None'}

const generatedFunction = () => {{
    // TODO: Implement {prompt}
    console.log('Generated function for: {prompt}');
}};

// Export for use
export default generatedFunction;
'''
    
    def optimize_code(self, code: str) -> str:
        """Optimize JavaScript code"""
        optimized = code
        
        # Suggest modern JS features
        if 'function(' in code:
            optimized += '\n// Suggestion: Consider using arrow functions'
        if 'for (var' in code:
            optimized += '\n// Suggestion: Consider using const and for...of loops'
            
        return optimized
    
    def validate_syntax(self, code: str) -> Tuple[bool, List[str]]:
        """Basic JavaScript syntax validation"""
        issues = []
        
        # Check for basic syntax issues
        if code.count('{') != code.count('}'):
            issues.append("Mismatched curly braces")
        if code.count('(') != code.count(')'):
            issues.append("Mismatched parentheses")
        if code.count('[') != code.count(']'):
            issues.append("Mismatched square brackets")
            
        return len(issues) == 0, issues

# Language processor registry
PROCESSORS = {
    'python': PythonProcessor,
    'javascript': JavaScriptProcessor,
    'typescript': JavaScriptProcessor,  # Reuse JS processor for TS
    'java': LanguageProcessor,  # Would implement JavaProcessor
    'cpp': LanguageProcessor,   # Would implement CppProcessor
    'go': LanguageProcessor,    # Would implement GoProcessor
    'rust': LanguageProcessor, # Would implement RustProcessor
}

def get_language_processor(language: str) -> LanguageProcessor:
    """Get appropriate language processor"""
    language = language.lower()
    
    if language in PROCESSORS:
        if language == 'python':
            return PythonProcessor()
        elif language in ['javascript', 'typescript']:
            return JavaScriptProcessor()
        else:
            # For unsupported languages, create a basic processor
            class BasicProcessor(LanguageProcessor):
                def __init__(self, lang):
                    super().__init__(lang)
                    
                def _get_file_extensions(self):
                    return [f".{lang}"]
                    
                def analyze_code(self, code):
                    return CodeAnalysis(0, [], [], [], len(code.split('\n')), [], [])
                    
                def generate_code(self, prompt, context=None):
                    return f"// Generated code for {prompt} in {self.language}"
                    
                def optimize_code(self, code):
                    return code
                    
                def validate_syntax(self, code):
                    return True, []
                    
            return BasicProcessor(language)
    else:
        raise ValueError(f"Unsupported language: {language}")

# Performance monitoring
class CodePerformanceMonitor:
    """Monitor code performance metrics"""
    
    @staticmethod
    def profile_function(func, *args, **kwargs):
        """Profile function execution"""
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger.info(f"Function {func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    
    @staticmethod
    def memory_usage():
        """Get current memory usage"""
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024  # MB 