import json
import sys

def compute_final_accuracy(json_path: str) -> None:
    # Load the JSON file (expects a list of dicts)
    with open(json_path, "r") as f:
        data = json.load(f)

    total_solved = 0
    total_executed = 0

    for entry in data:
        total_executed += 1

        # Prefer the explicit "Solved" flag if present
        solved_flag = entry.get("Solved")
        if isinstance(solved_flag, bool):
            if solved_flag:
                total_solved += 1
        else:
            # Fallback: compare Answer vs Attempt answer
            ans = str(entry.get("Answer", "")).strip()
            attempt = str(entry.get("Attempt answer", "")).strip()
            if ans == attempt:
                total_solved += 1

    if total_executed == 0:
        accuracy = 0.0
    else:
        accuracy = total_solved / total_executed

    print(f"Total solved: {total_solved}")
    print(f"Total executed: {total_executed}")
    print(f"Final accuracy: {accuracy:.4f}")  # 4 decimal places


if __name__ == "__main__":
    # Usage: python compute_accuracy.py results.json
    if len(sys.argv) < 2:
        print("Usage: python compute_accuracy.py <path_to_json_file>")
        sys.exit(1)

    json_path = sys.argv[1]
    compute_final_accuracy(json_path)
