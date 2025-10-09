from typing import Dict, Any, List
import re

from ..graph.node import Node
from .agent_registry import AgentRegistry
from GDesigner.llm import LLMRegistry
from GDesigner.prompt.prompt_set_registry import PromptSetRegistry

@AgentRegistry.register("TestGenerator")
class TestGenerator(Node):
    def __init__(self, id: str | None = None, role: str = None, domain: str = "", llm_name: str = ""):
        super().__init__(id, "TestGenerator", domain, llm_name)
        self.llm = LLMRegistry.get(llm_name, model_name=llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = "Test Case Generator"
        self.constraint = """You are a test case generator. Your job is to create comprehensive test cases for code.
        Focus on:
        1. Edge cases and boundary conditions
        2. Invalid inputs and error handling
        3. Normal use cases
        4. Stress testing scenarios
        
        Generate test cases that will help verify the correctness and robustness of the code."""
        
    async def _process_inputs(self, raw_inputs: Dict[str, str], spatial_info: Dict[str, Dict], temporal_info: Dict[str, Dict], **kwargs) -> List[Any]:
        system_prompt = self.constraint
        user_prompt = f"The task is: {raw_inputs['task']}\n\n"
        
        # Extract function signature from the task
        task_lines = raw_inputs['task'].split('\n')
        func_signature = None
        for line in task_lines:
            if line.strip().startswith('def '):
                func_signature = line.strip()
                break
        
        if func_signature:
            user_prompt += f"Function signature: {func_signature}\n\n"
        
        # Look for existing test cases in the task
        if 'assert ' in raw_inputs['task']:
            user_prompt += "Existing test cases in the task:\n"
            for line in task_lines:
                if 'assert ' in line:
                    user_prompt += f"  {line.strip()}\n"
            user_prompt += "\n"
        
        user_prompt += "Generate additional test cases that cover edge cases, boundary conditions, and potential error scenarios."
        
        return system_prompt, user_prompt
