import re


def svamp_data_process(dataset):
    """
    Convert raw SVAMP JSON/JSONL list into the format expected by the runner:
    each item -> {"task": <question_text>, "answer": <normalized_answer_string>}

    We try to be robust to different SVAMP formats:
      - Original SVAMP: keys like "Body", "Question", "Answer"
      - Variants: may use "question", "answer", "final_ans", "result", etc.
    """
    list_data_dict = []

    for data in dataset:
        # ---- Build the question text ----
        # Original SVAMP format
        body = data.get("Body", "") or data.get("body", "")
        q_part = data.get("Question", "") or data.get("question", "")

        if body and q_part:
            q = f"{body.strip()} {q_part.strip()}".strip()
        else:
            # Fallbacks / variants
            q = (
                q_part
                or body
                or data.get("problem", "")
                or data.get("text", "")
                or data.get("Task", "")
            ).strip()

        if not q:
            # Last resort: stringified dict (should basically never be needed)
            q = str(data).strip()

        # ---- Extract the answer ----
        # Try common fields in order of likelihood
        raw_answer = (
            data.get("answer", None)
            or data.get("Answer", None)
            or data.get("final_ans", None)
            or data.get("result", None)
        )

        if raw_answer is None:
            raw_answer = ""

        raw_answer = str(raw_answer).strip()

        # Remove commas in numbers like "1,000"
        raw_answer = raw_answer.replace(",", "")

        # Clean math/LaTeX-ish formatting if any (for robustness)
        raw_answer = _strip_string(raw_answer)

        # Strip trailing '.' if it's just a sentence-ending period
        if raw_answer.endswith("."):
            raw_answer = raw_answer[:-1]

        item = {
            "task": q,
            "answer": raw_answer,
        }
        list_data_dict.append(item)

    return list_data_dict


def svamp_get_predict(pred_str):
    """
    Extract a numeric prediction from an LLM's raw output string.

    Tries:
      - 'The answer is ...'
      - 'boxed{...}'
      - last number in the string (default)

    Returns a string of digits (possibly with minus sign handled earlier),
    or '0' if nothing is found.
    """
    if 'The answer is ' in pred_str:
        pred = pred_str.split('The answer is ')[-1].strip()
    elif 'the answer is ' in pred_str:
        pred = pred_str.split('the answer is ')[-1].strip()
    elif 'boxed' in pred_str:
        ans = pred_str.split('boxed')[-1]
        if ans and ans[0] == '{':
            stack = 1
            a = ''
            for c in ans[1:]:
                if c == '{':
                    stack += 1
                    a += c
                elif c == '}':
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split('$')[0].strip()
        a = _strip_string(a)
        pred = a
    else:
        # Fallback: last number in the string (supports optional decimal)
        pattern = r'-?\d*\.?\d+'
        matches = re.findall(pattern, pred_str)
        if matches:
            pred = matches[-1]
        else:
            pred = ''

    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]
        if pred and pred[-1] == "/":
            pred = pred[:-1]

    pred = _strip_string(pred)

    # Handle 'boxed' that might still remain
    if 'boxed' in pred:
        ans = pred.split('boxed')[-1]
        if ans and ans[0] == '{':
            stack = 1
            a = ''
            for c in ans[1:]:
                if c == '{':
                    stack += 1
                    a += c
                elif c == '}':
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split('$')[0].strip()
        a = _strip_string(a)
        pred = a

    # SVAMP answers are effectively integers; grab the last integer
    if pred.isdigit():
        return pred
    else:
        matches = re.findall(r'\d+', pred)
        return matches[-1] if matches else '0'


def svamp_check_correctness(pred, gt):
    """
    Compare predicted answer vs ground-truth for SVAMP.

    Both `pred` and `gt` are strings. We:
      - clean them
      - normalize numeric representation
      - compare as integers when possible
    """
    pred = str(pred).strip()
    gt = str(gt).strip()

    pred = pred.replace(",", "")
    gt = gt.replace(",", "")

    pred = _strip_string(pred)
    gt = _strip_string(gt)

    pred = delete_extra_zero(pred)
    gt = delete_extra_zero(gt)

    try:
        pred_val = int(float(pred))
        gt_val = int(float(gt))
        return pred_val == gt_val
    except Exception:
        # Fallback: direct string equality if something weird happens
        return pred == gt


# ========= Helper functions (copied/adapted from GSM8K utilities) =========

def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if not split:
            continue
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def delete_extra_zero(n):
    try:
        n = float(n)
    except Exception:
        # If it's not parseable, just return as-is
        return n
    # Convert back to a clean string without trailing zeros
    s = str(n)
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if not substr:
                continue
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a_int = int(a)
        b_int = int(b)
        assert string == "{}/{}".format(a_int, b_int)
        new_string = "\\frac{" + str(a_int) + "}{" + str(b_int) + "}"
        return new_string
    except Exception:
        return string


def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        if len(splits) == 2:
            return splits[0]
    return string


def _strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # remove leading tiny "k = " or "q = " etc.
    if len(string.split("=")) == 2:
        left, right = string.split("=")
        if len(left) <= 2:
            string = right

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> canonical form
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string
