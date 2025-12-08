import os
import torch
import asyncio
from typing import List, Union, Optional, Dict, Any
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

from .format import Message
from .llm import LLM
# from .llm_registry import LLMRegistry

load_dotenv()

# Configuration from environment
HF_MODEL_PATH = os.getenv("HF_MODEL_PATH", "Qwen/Qwen2.5-8B-Instruct")
HF_CACHE_DIR = os.getenv("HF_CACHE_DIR", "./hf_cache")
HF_DEVICE = os.getenv("HF_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)


class HFModelManager:
    """Singleton manager for HuggingFace models to avoid reloading"""
    _instance = None
    _models = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(HFModelManager, cls).__new__(cls)
        return cls._instance
    
    def get_model(self, model_name: str):
        """Load or retrieve cached model and tokenizer"""
        if model_name not in self._models:
            logger.info(f"Loading HuggingFace model: {model_name}")
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=HF_CACHE_DIR,
                trust_remote_code=True
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=HF_CACHE_DIR,
                torch_dtype=torch.float16,
                device_map=HF_DEVICE,
                trust_remote_code=True
            )
            
            # Ensure padding token is set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            self._models[model_name] = {
                'tokenizer': tokenizer,
                'model': model
            }
            logger.info(f"Model {model_name} loaded successfully on {HF_DEVICE}")
        
        return self._models[model_name]


async def hf_chat(
    model_name: str,
    messages: List[Dict],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = 0.2,
    num_comps: Optional[int] = 1,
):
    """
    HuggingFace chat completion function
    """
    manager = HFModelManager()
    model_data = manager.get_model(model_name)
    tokenizer = model_data['tokenizer']
    model = model_data['model']
    
    # Convert messages to chat format
    formatted_messages = []
    for msg in messages:
        if isinstance(msg, dict):
            formatted_messages.append(msg)
        elif hasattr(msg, 'to_dict'):
            formatted_messages.append(msg.to_dict())
        else:
            formatted_messages.append({
                'role': getattr(msg, 'role', 'user'),
                'content': getattr(msg, 'content', str(msg))
            })
    
    # ADD THIS: For code generation tasks, prepend a strong system message
    first_content = formatted_messages[0].get('content', '') if formatted_messages else ''
    if 'def ' in first_content and 'docstring' in first_content.lower():
        # This is a code generation task
        code_system_msg = {
            'role': 'system',
            'content': (
                "You are a code generation assistant. "
                "Generate ONLY executable Python code. "
                "Do NOT use <think> tags, explanations, or comments outside the code. "
                "Start directly with 'def' and write complete, runnable code."
            )
        }
        formatted_messages = [code_system_msg] + formatted_messages
    
    # Apply chat template
    try:
        prompt = tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception as e:
        logger.warning(f"Chat template failed: {e}. Using fallback formatting.")
        prompt = ""
        for msg in formatted_messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'system':
                prompt += f"System: {content}\n\n"
            elif role == 'user':
                prompt += f"User: {content}\n\n"
            elif role == 'assistant':
                prompt += f"Assistant: {content}\n\n"
        prompt += "Assistant: "
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    max_new_tokens = max_tokens if max_tokens else 512
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated part (not the prompt)
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return response


# @LLMRegistry.register('HFChat')
class HFChat(LLM):
    """HuggingFace Chat LLM implementation"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        logger.info(f"Initialized HFChat with model: {model_name}")
    
    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        """Async generation method"""
        
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS
        
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        
        return await hf_chat(self.model_name, messages, max_tokens, temperature, num_comps)
    
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        """Synchronous generation (wraps async)"""
        return asyncio.run(self.agen(messages, max_tokens, temperature, num_comps))