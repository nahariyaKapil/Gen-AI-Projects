"""
Language-specific processors for code analysis and generation.
Handles syntax validation, function extraction, and style guides.
"""

import ast
import re
import subprocess
import tempfile
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json

# Expert-level configuration
class ExpertLogger:
    def info(self, message):
        print(f"[INFO] {message}")
    
    def error(self, message):
        print(f"[ERROR] {message}")
    
    def warning(self, message):
        print(f"[WARNING] {message}")

logger = ExpertLogger()

# Forward declaration - alias will be defined after BaseLanguageProcessor
LanguageProcessor = None


@dataclass
class FunctionInfo:
    """Information about a function extracted from code."""
    name: str
    parameters: List[str]
    return_type: Optional[str]
    docstring: Optional[str]
    line_start: int
    line_end: int
    complexity: int = 0


@dataclass
class CodeValidationResult:
    """Result of code validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]


class BaseLanguageProcessor(ABC):
    """Base class for language-specific processors."""
    
    def __init__(self, language: str):
        self.language = language
        self.style_guide = self._load_style_guide()
    
    @abstractmethod
    def validate_syntax(self, code: str) -> CodeValidationResult:
        """Validate code syntax."""
        pass
    
    @abstractmethod
    def extract_functions(self, code: str) -> List[FunctionInfo]:
        """Extract function information from code."""
        pass
    
    @abstractmethod
    def format_code(self, code: str) -> str:
        """Format code according to style guide."""
        pass
    
    @abstractmethod
    def _load_style_guide(self) -> Dict[str, Any]:
        """Load language-specific style guide."""
        pass
    
    def get_suggestions(self, code: str) -> List[str]:
        """Get improvement suggestions for code."""
        suggestions = []
        
        # Common suggestions
        if len(code.split('\n')) > 100:
            suggestions.append("Consider breaking this into smaller functions")
        
        if 'TODO' in code or 'FIXME' in code:
            suggestions.append("Address TODO/FIXME comments")
        
        # Check for hardcoded values
        if re.search(r'\b\d{3,}\b', code):
            suggestions.append("Consider using constants for magic numbers")
        
        return suggestions


# Set the alias after BaseLanguageProcessor is defined
LanguageProcessor = BaseLanguageProcessor


class PythonProcessor(BaseLanguageProcessor):
    """Processor for Python code."""
    
    def __init__(self):
        super().__init__("python")
    
    def _load_style_guide(self) -> Dict[str, Any]:
        """Load PEP 8 style guide rules."""
        return {
            "max_line_length": 88,  # Black default
            "indent_size": 4,
            "naming_conventions": {
                "function": "snake_case",
                "variable": "snake_case",
                "class": "PascalCase",
                "constant": "UPPER_CASE"
            },
            "imports": {
                "stdlib_first": True,
                "separate_groups": True,
                "no_wildcard": True
            }
        }
    
    def validate_syntax(self, code: str) -> CodeValidationResult:
        """Validate Python syntax using AST."""
        errors = []
        warnings = []
        suggestions = []
        
        try:
            # Parse AST
            tree = ast.parse(code)
            
            # Additional style checks
            lines = code.split('\n')
            for i, line in enumerate(lines, 1):
                # Line length check
                if len(line) > self.style_guide["max_line_length"]:
                    warnings.append(f"Line {i}: Line too long ({len(line)} > {self.style_guide['max_line_length']})")
                
                # Indentation check
                if line.strip() and not line.startswith('#'):
                    leading_spaces = len(line) - len(line.lstrip(' '))
                    if leading_spaces % self.style_guide["indent_size"] != 0:
                        warnings.append(f"Line {i}: Indentation not multiple of {self.style_guide['indent_size']}")
            
            # Check imports
            imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
            if any(isinstance(node, ast.ImportFrom) and node.module == '*' for node in imports):
                warnings.append("Avoid wildcard imports")
            
            # Check for naming conventions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if not re.match(r'^[a-z_][a-z0-9_]*$', node.name):
                        warnings.append(f"Function '{node.name}' should use snake_case")
                elif isinstance(node, ast.ClassDef):
                    if not re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
                        warnings.append(f"Class '{node.name}' should use PascalCase")
            
            return CodeValidationResult(True, errors, warnings, suggestions)
            
        except SyntaxError as e:
            errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
            return CodeValidationResult(False, errors, warnings, suggestions)
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            return CodeValidationResult(False, errors, warnings, suggestions)
    
    def extract_functions(self, code: str) -> List[FunctionInfo]:
        """Extract Python functions using AST."""
        functions = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Extract parameters
                    params = []
                    for arg in node.args.args:
                        param_str = arg.arg
                        if arg.annotation:
                            param_str += f": {ast.unparse(arg.annotation)}"
                        params.append(param_str)
                    
                    # Extract return type
                    return_type = None
                    if node.returns:
                        return_type = ast.unparse(node.returns)
                    
                    # Extract docstring
                    docstring = None
                    if (node.body and isinstance(node.body[0], ast.Expr) and
                        isinstance(node.body[0].value, ast.Constant) and
                        isinstance(node.body[0].value.value, str)):
                        docstring = node.body[0].value.value
                    
                    # Calculate complexity (simplified)
                    complexity = self._calculate_complexity(node)
                    
                    functions.append(FunctionInfo(
                        name=node.name,
                        parameters=params,
                        return_type=return_type,
                        docstring=docstring,
                        line_start=node.lineno,
                        line_end=node.end_lineno or node.lineno,
                        complexity=complexity
                    ))
        
        except Exception as e:
            logger.error(f"Error extracting Python functions: {e}")
        
        return functions
    
    def format_code(self, code: str) -> str:
        """Format Python code using Black-style formatting."""
        try:
            # Use black if available
            import black
            return black.format_str(code, mode=black.FileMode())
        except ImportError:
            logger.warning("Black not available, using basic formatting")
            return self._basic_python_format(code)
        except Exception as e:
            logger.error(f"Error formatting Python code: {e}")
            return code
    
    def _basic_python_format(self, code: str) -> str:
        """Basic Python formatting."""
        lines = code.split('\n')
        formatted_lines = []
        indent_level = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                formatted_lines.append('')
                continue
            
            # Adjust indent for closing brackets/keywords
            if stripped.startswith(('elif', 'else', 'except', 'finally')):
                current_indent = max(0, indent_level - 1)
            elif stripped.startswith((')', '}', ']')):
                current_indent = max(0, indent_level - 1)
            else:
                current_indent = indent_level
            
            # Format the line
            formatted_line = '    ' * current_indent + stripped
            formatted_lines.append(formatted_line)
            
            # Adjust indent for next line
            if stripped.endswith(':'):
                indent_level += 1
            elif stripped.startswith(('return', 'break', 'continue', 'raise', 'pass')):
                indent_level = max(0, indent_level - 1)
        
        return '\n'.join(formatted_lines)
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity


class JavaScriptProcessor(BaseLanguageProcessor):
    """Processor for JavaScript code."""
    
    def __init__(self):
        super().__init__("javascript")
    
    def _load_style_guide(self) -> Dict[str, Any]:
        """Load JavaScript style guide (Airbnb-style)."""
        return {
            "max_line_length": 100,
            "indent_size": 2,
            "quotes": "single",
            "semicolons": True,
            "naming_conventions": {
                "function": "camelCase",
                "variable": "camelCase",
                "class": "PascalCase",
                "constant": "UPPER_CASE"
            }
        }
    
    def validate_syntax(self, code: str) -> CodeValidationResult:
        """Validate JavaScript syntax."""
        errors = []
        warnings = []
        suggestions = []
        
        # Basic syntax checks
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Line length
            if len(line) > self.style_guide["max_line_length"]:
                warnings.append(f"Line {i}: Line too long ({len(line)} > {self.style_guide['max_line_length']})")
            
            # Check for var usage
            if re.search(r'\bvar\s+', stripped):
                suggestions.append(f"Line {i}: Consider using 'let' or 'const' instead of 'var'")
            
            # Check for == usage
            if '==' in stripped and '===' not in stripped:
                suggestions.append(f"Line {i}: Consider using '===' for strict equality")
            
            # Semicolon check
            if (stripped and not stripped.startswith('//') and not stripped.startswith('/*') and
                not stripped.endswith((';', '{', '}', ')', ']')) and
                not re.match(r'^\s*(if|else|for|while|switch|function|class)', stripped)):
                warnings.append(f"Line {i}: Missing semicolon")
        
        # Check for balanced brackets
        brackets = {'(': 0, '[': 0, '{': 0}
        for char in code:
            if char in brackets:
                brackets[char] += 1
            elif char == ')':
                brackets['('] -= 1
            elif char == ']':
                brackets['['] -= 1
            elif char == '}':
                brackets['{'] -= 1
        
        for bracket, count in brackets.items():
            if count != 0:
                errors.append(f"Unbalanced brackets: {bracket}")
        
        is_valid = len(errors) == 0
        return CodeValidationResult(is_valid, errors, warnings, suggestions)
    
    def extract_functions(self, code: str) -> List[FunctionInfo]:
        """Extract JavaScript functions using regex patterns."""
        functions = []
        
        # Regular function declarations
        func_pattern = r'function\s+(\w+)\s*\(([^)]*)\)\s*{'
        for match in re.finditer(func_pattern, code):
            name = match.group(1)
            params_str = match.group(2).strip()
            params = [p.strip() for p in params_str.split(',')] if params_str else []
            
            line_num = code[:match.start()].count('\n') + 1
            
            functions.append(FunctionInfo(
                name=name,
                parameters=params,
                return_type=None,
                docstring=None,
                line_start=line_num,
                line_end=line_num,  # Simplified
                complexity=1
            ))
        
        # Arrow functions
        arrow_pattern = r'const\s+(\w+)\s*=\s*\(([^)]*)\)\s*=>'
        for match in re.finditer(arrow_pattern, code):
            name = match.group(1)
            params_str = match.group(2).strip()
            params = [p.strip() for p in params_str.split(',')] if params_str else []
            
            line_num = code[:match.start()].count('\n') + 1
            
            functions.append(FunctionInfo(
                name=name,
                parameters=params,
                return_type=None,
                docstring=None,
                line_start=line_num,
                line_end=line_num,
                complexity=1
            ))
        
        return functions
    
    def format_code(self, code: str) -> str:
        """Basic JavaScript formatting."""
        lines = code.split('\n')
        formatted_lines = []
        indent_level = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                formatted_lines.append('')
                continue
            
            # Adjust indent for closing brackets
            if stripped.startswith(('}', ')', ']')):
                current_indent = max(0, indent_level - 1)
            else:
                current_indent = indent_level
            
            # Format the line
            formatted_line = '  ' * current_indent + stripped
            formatted_lines.append(formatted_line)
            
            # Adjust indent for next line
            if stripped.endswith(('{', '(', '[')):
                indent_level += 1
            elif stripped.startswith(('}', ')', ']')):
                indent_level = max(0, indent_level - 1)
        
        return '\n'.join(formatted_lines)


class TypeScriptProcessor(JavaScriptProcessor):
    """Processor for TypeScript code (extends JavaScript)."""
    
    def __init__(self):
        super().__init__()
        self.language = "typescript"
    
    def validate_syntax(self, code: str) -> CodeValidationResult:
        """Validate TypeScript syntax."""
        result = super().validate_syntax(code)
        
        # Additional TypeScript-specific checks
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Check for any usage
            if re.search(r':\s*any\b', stripped):
                result.warnings.append(f"Line {i}: Avoid using 'any' type")
            
            # Check for type annotations
            if re.search(r'function\s+\w+\s*\([^)]*\)\s*{', stripped):
                if ':' not in stripped:
                    result.suggestions.append(f"Line {i}: Consider adding return type annotation")
        
        return result
    
    def extract_functions(self, code: str) -> List[FunctionInfo]:
        """Extract TypeScript functions with type information."""
        functions = super().extract_functions(code)
        
        # Enhanced TypeScript function pattern with types
        ts_func_pattern = r'function\s+(\w+)\s*\(([^)]*)\)\s*:\s*([^{]+)\s*{'
        for match in re.finditer(ts_func_pattern, code):
            name = match.group(1)
            params_str = match.group(2).strip()
            return_type = match.group(3).strip()
            
            params = [p.strip() for p in params_str.split(',')] if params_str else []
            line_num = code[:match.start()].count('\n') + 1
            
            # Update existing function or add new one
            existing = next((f for f in functions if f.name == name), None)
            if existing:
                existing.return_type = return_type
            else:
                functions.append(FunctionInfo(
                    name=name,
                    parameters=params,
                    return_type=return_type,
                    docstring=None,
                    line_start=line_num,
                    line_end=line_num,
                    complexity=1
                ))
        
        return functions


class JavaProcessor(BaseLanguageProcessor):
    """Processor for Java code."""
    
    def __init__(self):
        super().__init__("java")
    
    def _load_style_guide(self) -> Dict[str, Any]:
        """Load Java style guide (Google Java Style)."""
        return {
            "max_line_length": 100,
            "indent_size": 2,
            "naming_conventions": {
                "method": "camelCase",
                "variable": "camelCase",
                "class": "PascalCase",
                "constant": "UPPER_CASE",
                "package": "lowercase"
            },
            "braces": "new_line"
        }
    
    def validate_syntax(self, code: str) -> CodeValidationResult:
        """Validate Java syntax."""
        errors = []
        warnings = []
        suggestions = []
        
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Line length
            if len(line) > self.style_guide["max_line_length"]:
                warnings.append(f"Line {i}: Line too long ({len(line)} > {self.style_guide['max_line_length']})")
            
            # Check for naming conventions
            if re.search(r'class\s+([a-z][a-zA-Z0-9]*)', stripped):
                warnings.append(f"Line {i}: Class names should start with uppercase")
            
            # Check for proper imports
            if stripped.startswith('import') and '.*' in stripped:
                suggestions.append(f"Line {i}: Avoid wildcard imports")
        
        # Check for balanced brackets
        brackets = {'(': 0, '[': 0, '{': 0}
        for char in code:
            if char in brackets:
                brackets[char] += 1
            elif char == ')':
                brackets['('] -= 1
            elif char == ']':
                brackets['['] -= 1
            elif char == '}':
                brackets['{'] -= 1
        
        for bracket, count in brackets.items():
            if count != 0:
                errors.append(f"Unbalanced brackets: {bracket}")
        
        is_valid = len(errors) == 0
        return CodeValidationResult(is_valid, errors, warnings, suggestions)
    
    def extract_functions(self, code: str) -> List[FunctionInfo]:
        """Extract Java methods."""
        functions = []
        
        # Method pattern
        method_pattern = r'(public|private|protected)?\s*(static)?\s*(\w+)\s+(\w+)\s*\(([^)]*)\)\s*{'
        for match in re.finditer(method_pattern, code):
            visibility = match.group(1) or 'package'
            static = match.group(2) or ''
            return_type = match.group(3)
            name = match.group(4)
            params_str = match.group(5).strip()
            
            params = []
            if params_str:
                for param in params_str.split(','):
                    param = param.strip()
                    if param:
                        params.append(param)
            
            line_num = code[:match.start()].count('\n') + 1
            
            functions.append(FunctionInfo(
                name=name,
                parameters=params,
                return_type=return_type,
                docstring=None,
                line_start=line_num,
                line_end=line_num,
                complexity=1
            ))
        
        return functions
    
    def format_code(self, code: str) -> str:
        """Basic Java formatting."""
        lines = code.split('\n')
        formatted_lines = []
        indent_level = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                formatted_lines.append('')
                continue
            
            # Adjust indent for closing brackets
            if stripped.startswith('}'):
                current_indent = max(0, indent_level - 1)
            else:
                current_indent = indent_level
            
            # Format the line
            formatted_line = '  ' * current_indent + stripped
            formatted_lines.append(formatted_line)
            
            # Adjust indent for next line
            if stripped.endswith('{'):
                indent_level += 1
            elif stripped.startswith('}'):
                indent_level = max(0, indent_level - 1)
        
        return '\n'.join(formatted_lines)


class LanguageProcessorFactory:
    """Factory for creating language processors."""
    
    _processors = {
        'python': PythonProcessor,
        'javascript': JavaScriptProcessor,
        'typescript': TypeScriptProcessor,
        'java': JavaProcessor,
    }
    
    @classmethod
    def create_processor(cls, language: str) -> Optional[BaseLanguageProcessor]:
        """Create a processor for the specified language."""
        language = language.lower()
        
        if language in cls._processors:
            return cls._processors[language]()
        
        logger.warning(f"No processor available for language: {language}")
        return None
    
    @classmethod
    def get_supported_languages(cls) -> List[str]:
        """Get list of supported languages."""
        return list(cls._processors.keys())
    
    @classmethod
    def register_processor(cls, language: str, processor_class: type):
        """Register a new language processor."""
        cls._processors[language.lower()] = processor_class


# Additional processors for other languages
class CppProcessor(BaseLanguageProcessor):
    """Basic C++ processor."""
    
    def __init__(self):
        super().__init__("cpp")
    
    def _load_style_guide(self) -> Dict[str, Any]:
        return {
            "max_line_length": 100,
            "indent_size": 2,
            "naming_conventions": {
                "function": "camelCase",
                "variable": "camelCase",
                "class": "PascalCase",
                "constant": "UPPER_CASE"
            }
        }
    
    def validate_syntax(self, code: str) -> CodeValidationResult:
        """Basic C++ validation."""
        errors = []
        warnings = []
        suggestions = []
        
        # Check for balanced brackets
        brackets = {'(': 0, '[': 0, '{': 0}
        for char in code:
            if char in brackets:
                brackets[char] += 1
            elif char == ')':
                brackets['('] -= 1
            elif char == ']':
                brackets['['] -= 1
            elif char == '}':
                brackets['{'] -= 1
        
        for bracket, count in brackets.items():
            if count != 0:
                errors.append(f"Unbalanced brackets: {bracket}")
        
        return CodeValidationResult(len(errors) == 0, errors, warnings, suggestions)
    
    def extract_functions(self, code: str) -> List[FunctionInfo]:
        """Extract C++ functions."""
        functions = []
        
        # Function pattern
        func_pattern = r'(\w+)\s+(\w+)\s*\(([^)]*)\)\s*{'
        for match in re.finditer(func_pattern, code):
            return_type = match.group(1)
            name = match.group(2)
            params_str = match.group(3).strip()
            
            params = [p.strip() for p in params_str.split(',')] if params_str else []
            line_num = code[:match.start()].count('\n') + 1
            
            functions.append(FunctionInfo(
                name=name,
                parameters=params,
                return_type=return_type,
                docstring=None,
                line_start=line_num,
                line_end=line_num,
                complexity=1
            ))
        
        return functions
    
    def format_code(self, code: str) -> str:
        """Basic C++ formatting."""
        return self._basic_c_style_format(code)
    
    def _basic_c_style_format(self, code: str) -> str:
        """Basic C-style formatting."""
        lines = code.split('\n')
        formatted_lines = []
        indent_level = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                formatted_lines.append('')
                continue
            
            if stripped.startswith('}'):
                current_indent = max(0, indent_level - 1)
            else:
                current_indent = indent_level
            
            formatted_line = '  ' * current_indent + stripped
            formatted_lines.append(formatted_line)
            
            if stripped.endswith('{'):
                indent_level += 1
            elif stripped.startswith('}'):
                indent_level = max(0, indent_level - 1)
        
        return '\n'.join(formatted_lines)


# Register additional processors
LanguageProcessorFactory.register_processor('cpp', CppProcessor)
LanguageProcessorFactory.register_processor('c++', CppProcessor)


def get_processor(language: str) -> Optional[BaseLanguageProcessor]:
    """Convenience function to get a language processor."""
    return LanguageProcessorFactory.create_processor(language)


def get_supported_languages() -> List[str]:
    """Get list of all supported languages."""
    return LanguageProcessorFactory.get_supported_languages()


def get_language_processor(language: str) -> Optional[BaseLanguageProcessor]:
    """Get language processor for specified language"""
    return LanguageProcessorFactory.create_processor(language) 