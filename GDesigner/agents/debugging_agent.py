from typing import Dict, Any, List
import re

from ..graph.node import Node
from .agent_registry import AgentRegistry
from GDesigner.llm import LLMRegistry
from GDesigner.prompt.prompt_set_registry import PromptSetRegistry

@AgentRegistry.register("DebuggingAgent")
class DebuggingAgent(Node):
    def __init__(self, id: str | None = None, role: str = None, domain: str = "", llm_name: str = ""):
        super().__init__(id, "DebuggingAgent", domain, llm_name)
        self.llm = LLMRegistry.get(llm_name, model_name=llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = "Debugging Specialist"
        self.constraint = """You are a debugging specialist. Your job is to analyze code that has issues and provide fixes.
        Focus on:
        1. Identifying syntax errors
        2. Finding logical bugs
        3. Suggesting step-by-step fixes
        4. Explaining why the code doesn't work
        
        Provide clear debugging guidance and corrected code."""
        
    async def _process_inputs(self, raw_inputs: Dict[str, str], spatial_info: Dict[str, Dict], temporal_info: Dict[str, Dict], **kwargs) -> List[Any]:
        system_prompt = self.constraint
        user_prompt = f"The task is: {raw_inputs['task']}\n\n"
        
        # Look for problematic code from other agents
        spatial_str = ""
        for id, info in spatial_info.items():
            if 'Error' in info['output'] or 'failed' in info['output'].lower() or 'bug' in info['output'].lower():
                spatial_str += f"Problematic code from Agent {id} ({info['role']}):\n{info['output']}\n\n"
        
        temporal_str = ""
        for id, info in temporal_info.items():
            if 'Error' in info['output'] or 'failed' in info['output'].lower() or 'bug' in info['output'].lower():
                temporal_str += f"Previous problematic code from Agent {id} ({info['role']}):\n{info['output']}\n\n"
        
        if spatial_str or temporal_str:
            user_prompt += f"Please analyze and debug the following code:\n\n{spatial_str}{temporal_str}"
            user_prompt += "Identify the issues and provide corrected code with explanations."
        else:
            user_prompt += "Review the task and identify potential issues that might arise in the implementation. Provide debugging guidance."
        
        return system_prompt, user_prompt
