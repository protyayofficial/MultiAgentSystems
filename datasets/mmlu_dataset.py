import re
from typing import List, Dict, Any, Union

def mmlu_data_process(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    processed_dataset = []
    for record in dataset:
        question = record["question"]
        choices = record["choices"]
        
        formatted_choices = "\\n".join([f"{i}. {choice}" for i, choice in enumerate(choices)])
        task = f"Question: {question}\\n\\nChoices:\\n{formatted_choices}\\n\\nProvide the index of the correct answer."
        
        processed_dataset.append({
            "task": task,
            "choices": choices,
            "answer": record["answer"] # Storing the index as the answer
        })
    return processed_dataset

def mmlu_get_predict(model_response: str, choices: List[str]) -> Union[int, None]:
    numbers = re.findall(r'\b(\d+)\b', model_response)
    if numbers:
        try:
            predicted_index = int(numbers[-1])
            if 0 <= predicted_index < len(choices):
                return predicted_index
        except (ValueError, IndexError):
            pass

    for i, choice in enumerate(choices):
        if choice.lower() in model_response.lower():
            return i
            
    # Handle letter responses (A=0, B=1, C=2, D=3)
    letter_mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
    for letter, index in letter_mapping.items():
        if letter in model_response.upper():
            return index
    return None

def mmlu_check_correctness(predicted_answer: Union[int, None], ground_truth: int) -> bool:
    """
    Checks if the predicted answer matches the ground truth for MMLU dataset.
    """
    if predicted_answer is None:
        return False
    return predicted_answer == ground_truth
