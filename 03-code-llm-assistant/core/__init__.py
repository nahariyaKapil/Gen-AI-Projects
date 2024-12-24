"""
Code LLM Assistant Core Module

Advanced code generation and assistance system with real-time integration,
multi-language support, and intelligent code analysis capabilities.
"""

from .code_engine import CodeEngine
from .language_processors import LanguageProcessor, PythonProcessor, JavaScriptProcessor
from .code_analyzer import CodeAnalyzer
from .integration_manager import IntegrationManager
from .suggestion_engine import SuggestionEngine

__all__ = [
    'CodeEngine',
    'LanguageProcessor',
    'PythonProcessor', 
    'JavaScriptProcessor',
    'CodeAnalyzer',
    'IntegrationManager',
    'SuggestionEngine'
]

__version__ = "1.0.0" 