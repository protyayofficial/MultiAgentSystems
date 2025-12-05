import re


def math_data_process(dataset):
    """
    Convert raw MATH JSONL list into the format expected by the runner:
    each item -> {"task": <problem>, "answer": <normalized_answer_string>}

    The JSONL rows look like:
      {
        "problem": "...",
        "solution": "...",
        "answer": "\\left( 3, \\frac{\\pi}{2} \\right)",
        "subject": "...",
        "level": 2,
        "unique_id": "test/precalculus/807.json"
      }
    """
    list_data_dict = []
    for data in dataset:
        # Question text
        q = data.get("problem", "").strip()

        # Answer is in "answer" for MATH; add a few fallbacks just in case
        raw_answer = data.get("answer", data.get("final_ans", data.get("ans", "")))
        raw_answer = str(raw_answer).strip()

        # Remove commas in numbers like "1,000" (rare but harmless)
        raw_answer = raw_answer.replace(",", "")

        # Clean LaTeX/math formatting for robust comparison
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


def math_get_predict(pred_str):
    """
    Extract a *symbolic or numeric* prediction from an LLM's raw output string.

    MATH answers are often LaTeX, e.g. "\\left( 3, \\frac{\\pi}{2} \\right)" or "\\frac{1}{2}".
    We try, in order:
      - anything inside '\\boxed{...}'
      - content after 'The answer is' / 'the answer is'
      - last math expression between dollar signs '$...$'
      - otherwise, the whole string

    Returns a normalized string (via _strip_string), not necessarily numeric.
    """
    pred = pred_str

    # 1) Prefer boxed answers
    if "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if ans and ans[0] == "{":
            # Parse balanced braces: boxed{ ... }
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            # Fallback: take up to next '$' or end
            a = ans.split("$")[0].strip()
        pred = a

    # 2) Otherwise, look for "The answer is ..."
    elif "The answer is" in pred_str or "the answer is" in pred_str:
        lower = pred_str.lower()
        key = "the answer is"
        idx = lower.rfind(key)
        pred = pred_str[idx + len(key):].strip()

        # If there is inline LaTeX $...$, grab the last such block
        if "$" in pred:
            parts = pred.split("$")
            # odd indices 1,3,5,... are inside dollars
            math_chunks = [parts[i] for i in range(1, len(parts), 2)]
            if math_chunks:
                pred = math_chunks[-1].strip()

    # 3) Otherwise, try last $...$ block in the whole string
    elif "$" in pred_str:
        parts = pred_str.split("$")
        math_chunks = [parts[i] for i in range(1, len(parts), 2)]
        if math_chunks:
            pred = math_chunks[-1].strip()
        else:
            pred = pred_str

    # 4) Fallback: use the whole string
    else:
        pred = pred_str

    # Remove wrapping '$...$' if still present
    pred = pred.strip()
    if pred.startswith("$") and pred.endswith("$") and len(pred) >= 2:
        pred = pred[1:-1].strip()

    # Normalize LaTeX-ish expression
    pred = _strip_string(pred)

    # As a little extra robustness: if it's just a simple integer-like number,
    # normalize its representation (e.g. "02" -> "2", "0.0" -> "0")
    simple_num_match = re.fullmatch(r"-?\d+(\.\d+)?", pred)
    if simple_num_match:
        pred = delete_extra_zero(pred)

    return pred


def math_check_correctness(pred, gt):
    """
    Compare predicted answer vs ground-truth for MATH.

    Both `pred` and `gt` are strings. We:
      - clean them
      - normalize LaTeX-ish formatting
      - try numeric comparison if both look numeric
      - otherwise, fall back to string equality

    This is *not* a full symbolic equivalence check (no SymPy, etc.),
    but it covers exact matches and simple numeric equivalences.
    """
    pred = str(pred).strip()
    gt = str(gt).strip()

    # Remove commas in large numbers, if any
    pred = pred.replace(",", "")
    gt = gt.replace(",", "")

    # Normalize LaTeX / formatting
    pred = _strip_string(pred)
    gt = _strip_string(gt)

    # Try numeric comparison when both look numeric
    if _looks_numeric(pred) and _looks_numeric(gt):
        try:
            pred_val = float(pred)
            gt_val = float(gt)
            return pred_val == gt_val
        except Exception:
            pass

    # Fallback: direct string equality
    return pred == gt


def _looks_numeric(s: str) -> bool:
    """Heuristic: does the string look like a plain number?"""
    return bool(re.fullmatch(r"-?\d+(\.\d+)?", s))


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
    # X/Y -> \frac{X}{Y}, but only in simple integer cases
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

    # remove dollar signs (escaped)
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
