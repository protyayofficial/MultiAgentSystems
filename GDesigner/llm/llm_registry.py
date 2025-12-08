from class_registry import ClassRegistry

# Create base registry
_base_registry = ClassRegistry()

# Import both backends
from .gpt_chat import GPTChat
from .hf_chat import HFChat


def get_llm_backend(model_name: str):
    """
    Determine which LLM backend to use based on model name
    
    OpenAI models: gpt-4, gpt-4o, gpt-3.5-turbo, o1-, etc.
    HuggingFace models: Qwen/*, meta-llama/*, deepseek-ai/*, etc.
    """
    openai_prefixes = ['gpt-', 'o1-', 'text-']
    
    if any(model_name.startswith(prefix) for prefix in openai_prefixes):
        return 'GPTChat'
    else:
        return 'HFChat'


class SmartLLMRegistry:
    """
    Wrapper around ClassRegistry that automatically detects backend
    """
    
    def __init__(self, base_registry):
        self._registry = base_registry
        # Register both backends
        self._registry.register('GPTChat')(GPTChat)
        self._registry.register('HFChat')(HFChat)
    
    def get(self, key, *args, **kwargs):
        """
        Smart get that detects backend from model name
        
        Args:
            key: Either backend name ('GPTChat', 'HFChat') or model name ('gpt-4', 'Qwen/...')
            *args, **kwargs: Passed to backend constructor
        """
        # Check if key is already a registered backend
        if key in ['GPTChat', 'HFChat']:
            backend_name = key
        else:
            # Detect backend from model name
            backend_name = get_llm_backend(key)
        
        # Get the backend class
        backend_class = self._registry.get_class(backend_name)
        
        # Instantiate with model_name
        # model_name might be in kwargs already, or we use key
        if 'model_name' not in kwargs:
            kwargs['model_name'] = key
        
        return self._registry.create_instance(backend_class, *args, **kwargs)
    
    def register(self, name):
        """Pass through to base registry"""
        return self._registry.register(name)
    
    def get_class(self, key):
        """Pass through to base registry"""
        return self._registry.get_class(key)


# Export the smart registry
LLMRegistry = SmartLLMRegistry(_base_registry)