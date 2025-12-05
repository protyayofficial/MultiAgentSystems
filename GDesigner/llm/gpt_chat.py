import aiohttp
import openai
import os
import time
import asyncio
from collections import deque
from typing import List, Union, Optional, Dict, Any
from tenacity import retry, wait_random_exponential, stop_after_attempt
from dotenv import load_dotenv

from .format import Message
from .price import cost_count
from .llm import LLM
from .llm_registry import LLMRegistry


OPENAI_API_KEYS = ['']
BASE_URL = ''

load_dotenv()
# Prefer custom names, fallback to standard OpenAI env names
_RAW_BASE_URL = os.getenv('BASE_URL') or os.getenv('OPENAI_API_BASE')
_RAW_API_KEY = os.getenv('API_KEY') or os.getenv('OPENAI_API_KEY')


def _build_chat_endpoint(base_url: Optional[str]) -> str:
    """Return a normalized chat completions endpoint.
    - If base_url already contains '/chat/completions', return it.
    - If base_url is like '.../v1', append '/chat/completions'.
    - If base_url is None/empty, use the default OpenAI endpoint.
    """
    default = 'https://api.openai.com/v1/chat/completions'
    if not base_url:
        return default
    base_url = base_url.rstrip('/')
    if base_url.endswith('/chat/completions'):
        return base_url
    # Common cases: 'https://api.openai.com/v1' or provider-compatible base
    return f"{base_url}/chat/completions"


MINE_BASE_URL = _build_chat_endpoint(_RAW_BASE_URL)
MINE_API_KEYS = _RAW_API_KEY

# =========================
# Client-side rate limiting
# =========================

# You can override these with environment variables if you want.
# Given your "Limit: 3, Used: 3" error, we default a bit lower than 3.
MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "2"))
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "2"))

_request_timestamps: deque = deque()
_request_lock = asyncio.Lock()
_concurrent_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


async def _wait_for_rpm_slot() -> None:
    """
    Simple sliding-window RPM limiter.
    Ensures we start at most MAX_REQUESTS_PER_MINUTE requests in any 60s window.
    """
    while True:
        async with _request_lock:
            now = time.monotonic()

            # Drop timestamps older than 60 seconds
            while _request_timestamps and now - _request_timestamps[0] >= 60.0:
                _request_timestamps.popleft()

            if len(_request_timestamps) < MAX_REQUESTS_PER_MINUTE:
                _request_timestamps.append(now)
                return

            oldest = _request_timestamps[0]
            sleep_for = max(0.0, 60.0 - (now - oldest))

        # Sleep *outside* the lock so other coroutines can also check
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)
        else:
            # Just in case; loop will re-check
            await asyncio.sleep(0)


@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=60))
async def achat(model_name: str, messages: List[Message]):
    request_url = MINE_BASE_URL
    authorization_key = MINE_API_KEYS
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {authorization_key}'
    }
    data: Dict[str, Any] = {
        "model": model_name,
        "messages": [m.to_dict() for m in messages],
        "stream": False,
    }

    # Enforce concurrency and RPM limits on the client side.
    async with _concurrent_semaphore:
        await _wait_for_rpm_slot()

        async with aiohttp.ClientSession() as session:
            async with session.post(request_url, headers=headers, json=data) as response:
                # Handle HTTP-level rate limits explicitly
                if response.status == 429:
                    # Respect Retry-After if present
                    retry_after = response.headers.get("Retry-After")
                    if retry_after is not None:
                        try:
                            await asyncio.sleep(float(retry_after))
                        except Exception:
                            # Best-effort only; ignore parse errors
                            pass
                    text = await response.text()
                    raise Exception(f"Rate limited by server (HTTP 429): {text[:200]}")

                # Retry transient server errors as well
                if response.status >= 500:
                    text = await response.text()
                    raise Exception(f"Server error {response.status}: {text[:200]}")

                # If provider returns HTML (e.g., 404 page), raise a clearer error
                if 'application/json' not in (response.headers.get('Content-Type') or ''):
                    text = await response.text()
                    raise aiohttp.ContentTypeError(
                        response.request_info,
                        history=response.history,
                        message=(
                            f"Unexpected content-type: {response.headers.get('Content-Type')} "
                            f"at {request_url}. Body preview: {text[:200]}"
                        )
                    )

                response_data = await response.json()

                # OpenAI / compatible error payloads
                if 'error' in response_data and 'choices' not in response_data:
                    err_msg = response_data['error'].get('message', 'Unknown error')
                    low = err_msg.lower()
                    if "rate limit" in low or "please try again in" in low or "limit:" in low:
                        # Trigger tenacity retry
                        raise Exception(f"OpenAI API Error (rate limit): {err_msg}")
                    raise Exception(f"OpenAI API Error: {err_msg}")

                if 'choices' not in response_data:
                    error_message = response_data.get('error', {}).get('message', 'Unknown error')
                    raise Exception(f"OpenAI API Error: {error_message}")

                prompt = "".join([item.content for item in messages])
                completion = response_data['choices'][0]['message']['content']
                cost_count(prompt, completion, model_name)
                return completion


@LLMRegistry.register('gpt-4o')
@LLMRegistry.register('gpt-4o-mini')
@LLMRegistry.register('gpt-4-turbo')
@LLMRegistry.register("gpt-4-0125-preview")
@LLMRegistry.register("gpt-4-1106-preview")
@LLMRegistry.register("gpt-4-vision-preview")
@LLMRegistry.register("gpt-4-0314")
@LLMRegistry.register("gpt-4-32k")
@LLMRegistry.register("gpt-4-32k-0314")
@LLMRegistry.register("gpt-4-0613")
@LLMRegistry.register("gpt-3.5-turbo-0125")
@LLMRegistry.register("gpt-3.5-turbo-1106")
@LLMRegistry.register("gpt-3.5-turbo-instruct")
@LLMRegistry.register("gpt-3.5-turbo-0301")
@LLMRegistry.register("gpt-3.5-turbo-0613")
@LLMRegistry.register("gpt-3.5-turbo-16k-0613")
@LLMRegistry.register("gpt-3.5-turbo")
@LLMRegistry.register("gpt-4")
@LLMRegistry.register("Qwen/Qwen3-8B")
class GPTChat(LLM):

    def __init__(self, model_name: str, temperature: float = 0.7, top_p: float = 1.0, max_tokens: int = 1024):
        self.model_name = model_name
        # You can store / use temperature, top_p, max_tokens if needed
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    async def agen(
        self,
        messages: Union[List[Message], List[Dict[str, Any]], str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = getattr(self, "DEFAULT_MAX_TOKENS", self.max_tokens)
        if temperature is None:
            temperature = getattr(self, "DEFAULT_TEMPERATURE", self.temperature)
        if num_comps is None:
            num_comps = getattr(self, "DEFUALT_NUM_COMPLETIONS", 1)

        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        elif isinstance(messages, list) and all(isinstance(m, dict) for m in messages):
            messages = [Message(role=m['role'], content=m['content']) for m in messages]

        # NOTE: achat currently ignores max_tokens / temperature / num_comps;
        # if you want them, you can extend `data` in `achat`.
        return await achat(self.model_name, messages)

    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        # If you use sync gen, you can wrap `agen` with `asyncio.run` or similar.
        raise NotImplementedError("Use `agen` (async) for now.")


# import aiohttp
# import openai
# import os
# import time
# from typing import List, Union, Optional
# from tenacity import retry, wait_random_exponential, stop_after_attempt
# from typing import Dict, Any
# from dotenv import load_dotenv

# from .format import Message
# from .price import cost_count
# from .llm import LLM
# from .llm_registry import LLMRegistry


# OPENAI_API_KEYS = ['']
# BASE_URL = ''

# load_dotenv()
# # Prefer custom names, fallback to standard OpenAI env names
# _RAW_BASE_URL = os.getenv('BASE_URL') or os.getenv('OPENAI_API_BASE')
# _RAW_API_KEY = os.getenv('API_KEY') or os.getenv('OPENAI_API_KEY')


# def _build_chat_endpoint(base_url: Optional[str]) -> str:
#     """Return a normalized chat completions endpoint.
#     - If base_url already contains '/chat/completions', return it.
#     - If base_url is like '.../v1', append '/chat/completions'.
#     - If base_url is None/empty, use the default OpenAI endpoint.
#     """
#     default = 'https://api.openai.com/v1/chat/completions'
#     if not base_url:
#         return default
#     base_url = base_url.rstrip('/')
#     if base_url.endswith('/chat/completions'):
#         return base_url
#     # Common cases: 'https://api.openai.com/v1' or provider-compatible base
#     return f"{base_url}/chat/completions"


# MINE_BASE_URL = _build_chat_endpoint(_RAW_BASE_URL)
# MINE_API_KEYS = _RAW_API_KEY


# @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=60))
# async def achat(model_name:str, messages:list):
#     request_url = MINE_BASE_URL
#     authorization_key = MINE_API_KEYS
#     headers = {
#         'Content-Type': 'application/json',
#         'Authorization': f'Bearer {authorization_key}'
#     }
#     data = {
#         "model": model_name,
#         "messages": [m.to_dict() for m in messages],
#         "stream": False,
#     }
#     async with aiohttp.ClientSession() as session:
#         async with session.post(request_url, headers=headers ,json=data) as response:
#             # If provider returns HTML (e.g., 404 page), raise a clearer error
#             if 'application/json' not in (response.headers.get('Content-Type') or ''):
#                 text = await response.text()
#                 raise aiohttp.ContentTypeError(
#                     response.request_info,
#                     history=response.history,
#                     message=f"Unexpected content-type: {response.headers.get('Content-Type')} at {request_url}. Body preview: {text[:200]}"
#                 )
#             response_data = await response.json()
#             if 'choices' not in response_data:
#                 error_message = response_data.get('error', {}).get('message', 'Unknown error')
#                 raise Exception(f"OpenAI API Error: {error_message}")
#             prompt = "".join([item.content for item in messages])
#             completion = response_data['choices'][0]['message']['content']
#             cost_count(prompt, completion, model_name)
#             return completion


# @LLMRegistry.register('gpt-4o')
# @LLMRegistry.register('gpt-4o-mini')
# @LLMRegistry.register('gpt-4-turbo')
# @LLMRegistry.register("gpt-4-0125-preview")
# @LLMRegistry.register("gpt-4-1106-preview")
# @LLMRegistry.register("gpt-4-vision-preview")
# @LLMRegistry.register("gpt-4-0314")
# @LLMRegistry.register("gpt-4-32k")
# @LLMRegistry.register("gpt-4-32k-0314")
# @LLMRegistry.register("gpt-4-0613")
# @LLMRegistry.register("gpt-3.5-turbo-0125")
# @LLMRegistry.register("gpt-3.5-turbo-1106")
# @LLMRegistry.register("gpt-3.5-turbo-instruct")
# @LLMRegistry.register("gpt-3.5-turbo-0301")
# @LLMRegistry.register("gpt-3.5-turbo-0613")
# @LLMRegistry.register("gpt-3.5-turbo-16k-0613")
# @LLMRegistry.register("gpt-3.5-turbo")
# @LLMRegistry.register("gpt-4")
# class GPTChat(LLM):

#     def __init__(self, model_name: str, temperature: float = 0.7, top_p: float = 1.0, max_tokens: int = 1024):
#         self.model_name = model_name

#     async def agen(
#         self,
#         messages: List[Message],
#         max_tokens: Optional[int] = None,
#         temperature: Optional[float] = None,
#         num_comps: Optional[int] = None,
#         ) -> Union[List[str], str]:

#         if max_tokens is None:
#             max_tokens = self.DEFAULT_MAX_TOKENS
#         if temperature is None:
#             temperature = self.DEFAULT_TEMPERATURE
#         if num_comps is None:
#             num_comps = self.DEFUALT_NUM_COMPLETIONS
        
#         if isinstance(messages, str):
#             messages = [Message(role="user", content=messages)]
#         elif isinstance(messages, list) and all(isinstance(m, dict) for m in messages):
#             messages = [Message(role=m['role'], content=m['content']) for m in messages]
            
#         return await achat(self.model_name,messages)
    
#     def gen(
#         self,
#         messages: List[Message],
#         max_tokens: Optional[int] = None,
#         temperature: Optional[float] = None,
#         num_comps: Optional[int] = None,
#     ) -> Union[List[str], str]:
#         pass