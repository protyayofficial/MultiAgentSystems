# Import all agents to ensure they are registered
from .analyze_agent import AnalyzeAgent
from .code_writing import CodeWriting
from .math_solver import MathSolver
from .adversarial_agent import AdversarialAgent
from .final_decision import FinalRefer, FinalDirect, FinalWriteCode, FinalMajorVote
from .agent_registry import AgentRegistry

__all__ = [
    'AnalyzeAgent',
    'CodeWriting', 
    'MathSolver',
    'AdversarialAgent',
    'FinalRefer',
    'FinalDirect',
    'FinalWriteCode',
    'FinalMajorVote',
    'AgentRegistry'
]

# Import new specialized agents
from .code_reviewer import CodeReviewer
from .test_generator import TestGenerator
from .debugging_agent import DebuggingAgent

# Add to __all__
__all__.extend([
    'CodeReviewer',
    'TestGenerator', 
    'DebuggingAgent'
])
