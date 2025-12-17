import re
import signal
import subprocess
import tempfile
from typing import Tuple


def humaneval_data_process(dataset):
    """Process HumanEval dataset into the required format."""
    list_data_dict = []
    for data in dataset:
        item = {
            "task": data.get("prompt", ""),
            "test": data.get("test", ""),
            "entry_point": data.get("entry_point", "")
        }
        list_data_dict.append(item)
    return list_data_dict


def humaneval_get_predict(response: str) -> str:
    """
    Extract Python code from the model's response.
    Handles markdown code blocks and plain code.
    """
    if not response:
        return ""
    
    # CRITICAL FIX 1: Strip <think> tags from Qwen Thinking models
    # Handle both complete tags and missing opening tags
    if '<think>' in response or '</think>' in response:
        # Remove complete <think>...</think> blocks
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        # Handle missing opening tag - take everything after </think>
        if '</think>' in response:
            parts = response.split('</think>')
            response = parts[-1] if parts else response
        # Clean up any remaining orphaned tags
        response = response.replace('<think>', '').replace('</think>', '')
        response = response.strip()
    
    # CRITICAL FIX 2: Extract imports BEFORE extracting code
    import_lines = []
    for line in response.split('\n'):
        stripped = line.strip()
        if stripped.startswith('import ') or stripped.startswith('from '):
            import_lines.append(line)
    
    # Try to extract code from markdown code blocks
    if "```python" in response:
        # Extract code between ```python and ```
        pattern = r"```python\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            code = matches[0].strip()
            # Prepend imports if any
            if import_lines:
                code = '\n'.join(import_lines) + '\n\n' + code
            return code
    elif "```" in response:
        # Extract code between ``` and ```
        pattern = r"```\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            code = matches[0].strip()
            # Prepend imports if any
            if import_lines:
                code = '\n'.join(import_lines) + '\n\n' + code
            return code
    
    # If no code blocks found, try to extract code after common prefixes
    if "Here is the solution:" in response or "Here's the solution:" in response:
        lines = response.split("\n")
        code_lines = []
        in_code = False
        for line in lines:
            if line.strip().startswith("def "):
                in_code = True
            if in_code:
                code_lines.append(line)
        if code_lines:
            code = "\n".join(code_lines).strip()
            # Prepend imports if any
            if import_lines:
                code = '\n'.join(import_lines) + '\n\n' + code
            return code
    
    # If all else fails, look for function definitions
    if "def " in response:
        # Find the first function definition and extract everything from there
        idx = response.find("def ")
        if idx != -1:
            code = response[idx:].strip()
            # Prepend imports if any
            if import_lines:
                code = '\n'.join(import_lines) + '\n\n' + code
            return code
    
    # Return the whole response as a last resort
    # Prepend imports if any
    if import_lines:
        response = '\n'.join(import_lines) + '\n\n' + response
    return response.strip()


def check_correctness(task_prompt: str, predicted_code: str, test_code: str, timeout: float = 3.0) -> Tuple[float, str]:
    """
    Check if the predicted code passes the test cases.
    
    Returns:
        (score, result_string): score is 1.0 if passed, 0.0 if failed
    """
    if not predicted_code or not predicted_code.strip():
        return 0.0, "No code generated"
    
    # CRITICAL FIX 3: Extract function name and create proper 'candidate' alias
    func_match = re.search(r'def\s+(\w+)\s*\(', predicted_code)
    if not func_match:
        return 0.0, "No function definition found in predicted code"
    
    func_name = func_match.group(1)
    
    # Extract imports from task_prompt
    import_lines = []
    for line in task_prompt.split('\n'):
        if line.strip().startswith('from ') or line.strip().startswith('import '):
            import_lines.append(line)
    
    imports = '\n'.join(import_lines) if import_lines else ""
    
    # Build complete test program with proper candidate alias
    full_code = f"""{imports}

{predicted_code}

# Create alias for HumanEval test harness
candidate = {func_name}

{test_code}

# Run the test
check(candidate)
"""
    
    # Run the code in a subprocess with timeout
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full_code)
            temp_file = f.name
        
        try:
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode == 0:
                return 1.0, "Passed"
            else:
                error_msg = result.stderr or result.stdout or "Unknown error"
                return 0.0, f"Failed: {error_msg[:200]}"
        finally:
            import os
            if os.path.exists(temp_file):
                os.unlink(temp_file)
                
    except subprocess.TimeoutExpired:
        return 0.0, "Timeout"
    except Exception as e:
        return 0.0, f"Error: {str(e)[:200]}"
# import re
# import signal
# import subprocess
# import tempfile
# from typing import Tuple


# def humaneval_data_process(dataset):
#     """Process HumanEval dataset into the required format."""
#     list_data_dict = []
#     for data in dataset:
#         item = {
#             "task": data.get("prompt", ""),
#             "test": data.get("test", ""),
#             "entry_point": data.get("entry_point", "")
#         }
#         list_data_dict.append(item)
#     return list_data_dict


# def humaneval_get_predict(response: str) -> str:
#     """
#     Extract Python code from the model's response.
#     Handles markdown code blocks and plain code.
#     """
#     if not response:
#         return ""
    
#     # Try to extract code from markdown code blocks
#     if "```python" in response:
#         # Extract code between ```python and ```
#         pattern = r"```python\n(.*?)```"
#         matches = re.findall(pattern, response, re.DOTALL)
#         if matches:
#             return matches[0].strip()
#     elif "```" in response:
#         # Extract code between ``` and ```
#         pattern = r"```\n(.*?)```"
#         matches = re.findall(pattern, response, re.DOTALL)
#         if matches:
#             return matches[0].strip()
    
#     # If no code blocks found, try to extract code after common prefixes
#     if "Here is the solution:" in response or "Here's the solution:" in response:
#         lines = response.split("\n")
#         code_lines = []
#         in_code = False
#         for line in lines:
#             if line.strip().startswith("def "):
#                 in_code = True
#             if in_code:
#                 code_lines.append(line)
#         if code_lines:
#             return "\n".join(code_lines).strip()
    
#     # If all else fails, look for function definitions
#     if "def " in response:
#         # Find the first function definition and extract everything from there
#         idx = response.find("def ")
#         if idx != -1:
#             return response[idx:].strip()
    
#     # Return the whole response as a last resort
#     return response.strip()


# def check_correctness(task_prompt: str, predicted_code: str, test_code: str, timeout: float = 3.0) -> Tuple[float, str]:
#     """
#     Check if the predicted code passes the test cases.
    
#     Returns:
#         (score, result_string): score is 1.0 if passed, 0.0 if failed
#     """
#     if not predicted_code or not predicted_code.strip():
#         return 0.0, "No code generated"
    
#     # Combine the task prompt (function signature), predicted code, and test code
#     full_code = f"{task_prompt}\n{predicted_code}\n{test_code}\n\ncheck(candidate)"
    
#     # Run the code in a subprocess with timeout
#     try:
#         with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
#             f.write(full_code)
#             temp_file = f.name
        
#         try:
#             result = subprocess.run(
#                 ['python', temp_file],
#                 capture_output=True,
#                 text=True,
#                 timeout=timeout
#             )
            
#             if result.returncode == 0:
#                 return 1.0, "Passed"
#             else:
#                 error_msg = result.stderr or result.stdout or "Unknown error"
#                 return 0.0, f"Failed: {error_msg[:200]}"
#         finally:
#             import os
#             if os.path.exists(temp_file):
#                 os.unlink(temp_file)
                
#     except subprocess.TimeoutExpired:
#         return 0.0, "Timeout"
#     except Exception as e:
#         return 0.0, f"Error: {str(e)[:200]}"

