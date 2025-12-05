import re
from typing import Tuple
def count_test_cases(test_code):
    """Count the number of test cases (assertions) in the test code."""
    try:
        import ast
        tree = ast.parse(test_code)
        assert_count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                assert_count += 1
        return assert_count
    except:
        return test_code.count('assert')


def humaneval_data_process(dataset: list) -> list:
    """
    Processes the raw Humaneval dataset.
    """
    processed_dataset = []
    for record in dataset:
        processed_dataset.append({
            "task": record["prompt"],
            "test": record["test"],
            "entry_point": record["entry_point"]
        })
    return processed_dataset

def humaneval_get_predict(model_response: str) -> str:
    """
    Extracts the Python code block from the model's response.
    It looks for a ```python ... ``` block and extracts the content.
    If not found, it assumes the entire response is the code.
    """
    if '```python' in model_response:
        match = re.search(r"```python\n(.*?)\n```", model_response, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    if '```' in model_response:
        match = re.search(r"```\n(.*?)\n```", model_response, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    # Fallback for code that isn't in a markdown block
    return model_response.strip()

def check_correctness(prompt: str, completion: str, test: str) -> Tuple[float, str]:
    """
    Evaluates the generated code against the provided test cases with partial credit.
    Returns a decimal score (0.0 to 1.0) based on the percentage of test cases that pass.
    """
    program = f"{prompt}\n{completion}\n{test}"
    try:
        exec_globals = {}
        exec(program, exec_globals)
        return 1.0, "All tests passed"
    except AssertionError as e:
        # Try to run individual test cases to see how many pass
        try:
            import ast
            tree = ast.parse(test)
            assert_count = 0
            passed_count = 0
            
            # Count total assertions
            for node in ast.walk(tree):
                if isinstance(node, ast.Assert):
                    assert_count += 1
            
            # Try to run individual assertions
            for node in ast.walk(tree):
                if isinstance(node, ast.Assert):
                    try:
                        # Create a minimal test environment
                        test_env = exec_globals.copy()
                        test_env['candidate'] = exec_globals.get('candidate', None)
                        
                        # Execute just this assertion
                        exec(compile(ast.Module([node], []), '<string>', 'exec'), test_env)
                        passed_count += 1
                    except:
                        pass
            
            if assert_count > 0:
                score = passed_count / assert_count
                return score, f"Partial credit: {passed_count}/{assert_count} tests passed"
            else:
                return 0.0, "No test cases found"
                
        except Exception as e:
            return 0.0, f"Failed to parse test cases: {e}"
    except Exception as e:
        return 0.0, f"Execution failed: {type(e).__name__}: {e}"
