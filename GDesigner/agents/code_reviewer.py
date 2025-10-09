from typing import Dict, Any, List
import re

from ..graph.node import Node
from .agent_registry import AgentRegistry
from GDesigner.llm import LLMRegistry
from GDesigner.prompt.prompt_set_registry import PromptSetRegistry

@AgentRegistry.register("CodeReviewer")
class CodeReviewer(Node):
    def __init__(self, id: str | None = None, role: str = None, domain: str = "", llm_name: str = ""):
        super().__init__(id, "CodeReviewer", domain, llm_name)
        self.llm = LLMRegistry.get(llm_name, model_name=llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = "Code Reviewer"
        self.constraint = """You are a code reviewer. Your job is to review code for correctness, efficiency, and best practices. 
        Focus on:
        1. Logic correctness
        2. Edge case handling
        3. Code clarity and readability
        4. Potential bugs or issues
        
        Provide constructive feedback and suggest improvements. Be specific about what could be better."""
        
    async def _process_inputs(self, raw_inputs: Dict[str, str], spatial_info: Dict[str, Dict], temporal_info: Dict[str, Dict], **kwargs) -> List[Any]:
        system_prompt = self.constraint
        user_prompt = f"The task is: {raw_inputs['task']}\n\n"
        
        # Collect code from other agents
        spatial_str = ""
        for id, info in spatial_info.items():
            if '```python' in info['output'] or 'def ' in info['output']:
                spatial_str += f"Code from Agent {id} ({info['role']}):\n{info['output']}\n\n"
        
        temporal_str = ""
        for id, info in temporal_info.items():
            if '```python' in info['output'] or 'def ' in info['output']:
                temporal_str += f"Previous code from Agent {id} ({info['role']}):\n{info['output']}\n\n"
        
        if spatial_str:
            user_prompt += f"Please review the following code:\n\n{spatial_str}"
        if temporal_str:
            user_prompt += f"Also consider this previous code:\n\n{temporal_str}"
        
        user_prompt += "\nProvide your code review with specific feedback and suggestions for improvement."
        
        return system_prompt, user_prompt
