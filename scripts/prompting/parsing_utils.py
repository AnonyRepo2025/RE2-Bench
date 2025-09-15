import json
import re
from json import JSONDecodeError
from typing import Tuple, Optional

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = "\n".join(s.splitlines()[1:])
        if s.rstrip().endswith("```"):
            s = "\n".join(s.splitlines()[:-1])
    return s.strip()

def _remove_hash_comments(s: str) -> str:
    """Remove # comments, but keep # inside double-quoted strings."""
    out = []
    for line in s.splitlines():
        buf, in_str, esc = [], False, False
        for ch in line:
            if esc:
                buf.append(ch); esc = False; continue
            if ch == "\\":
                buf.append(ch); esc = True; continue
            if ch == '"':
                in_str = not in_str; buf.append(ch); continue
            if ch == "#" and not in_str:
                break
            buf.append(ch)
        out.append("".join(buf))
    return "\n".join(out)

def _remove_trailing_commas(s: str) -> str:
    # remove trailing commas before } or ]
    return re.sub(r",(\s*[}\]])", r"\1", s)

def _not_in_string_mask(s: str):
    """Return a boolean list: True for positions outside double-quoted strings."""
    mask, in_str, esc = [True]*len(s), False, False
    for i, ch in enumerate(s):
        if esc:
            esc = False
            mask[i] = not in_str
            continue
        if ch == "\\":
            esc = True
            mask[i] = not in_str
            continue
        if ch == '"':
            in_str = not in_str
            mask[i] = not in_str
            continue
        mask[i] = not in_str
    return mask

def _find_value_bounds(doc: str, err_pos: int) -> Tuple[int, int]:
    """
    From an error position, find the nearest preceding ':' (outside strings)
    and compute the value [start, end) span following that colon.
    Value ends at the first comma or closing brace/bracket at zero nesting,
    respecting quotes and tolerating parentheses inside.
    """
    mask = _not_in_string_mask(doc)

    # find preceding colon outside strings
    i = err_pos
    while i >= 0:
        if mask[i] and doc[i] == ":":
            break
        i -= 1
    if i < 0:
        return (err_pos, err_pos)  # give up

    # skip spaces after colon to value start
    start = i + 1
    n = len(doc)
    while start < n and doc[start].isspace():
        start += 1

    # scan forward to find value end
    depth_brace = depth_brack = depth_paren = 0
    in_str = False
    esc = False
    j = start
    while j < n:
        ch = doc[j]
        if esc:
            esc = False
            j += 1
            continue
        if ch == "\\" and in_str:
            esc = True
            j += 1
            continue
        if ch == '"':
            in_str = not in_str
            j += 1
            continue
        if not in_str:
            if ch == "{": depth_brace += 1
            elif ch == "}":
                if depth_brace == 0 and depth_brack == 0 and depth_paren == 0:
                    break
                depth_brace = max(0, depth_brace-1)
            elif ch == "[": depth_brack += 1
            elif ch == "]":
                if depth_brace == 0 and depth_brack == 0 and depth_paren == 0:
                    break
                depth_brack = max(0, depth_brack-1)
            elif ch == "(": depth_paren += 1
            elif ch == ")": depth_paren = max(0, depth_paren-1)
            elif ch == "," and depth_brace == 0 and depth_brack == 0 and depth_paren == 0:
                break
        j += 1
    end = j  # exclusive
    return (start, end)

def _quote_span(doc: str, start: int, end: int) -> str:
    raw = doc[start:end].strip()
    quoted = json.dumps(raw)  # safe JSON string
    return doc[:start] + quoted + doc[end:]

def _content_between_output_tags(text: str, tag: str) -> Optional[str]:
    """
    Return the content between [OUTPUT]/[INPUT] and [/OUTPUT]/[/OUTPUT] if both exist; else,
    return everything after [OUTPUT]/[INPUT]. Then, if a fenced block exists inside,
    return that fenced block; otherwise return the raw trimmed content.
    """
    if tag == "output":
        m = re.search(r'\[OUTPUT\](.*?)(?:\[/OUTPUT\]|$)', text, flags=re.DOTALL | re.IGNORECASE)
    else:
        m = re.search(r'\[INPUT\](.*?)(?:\[/INPUT\]|$)', text, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    tail = m.group(1)

    fence = re.search(r"```(.*?)```", tail, flags=re.DOTALL)
    if fence:
        return fence.group(0)
    return tail.strip()

def extract_output_json_easy(text: str, tag: str) -> dict:
    block = _content_between_output_tags(text, tag)
    if block is None:
        return {"json": None, "json_text": "", "error": f"Marker [{tag}] not found."}

    doc = _strip_code_fences(block)
    doc = _remove_hash_comments(doc)
    doc = _remove_trailing_commas(doc)

    max_passes = 25
    last_error = ""
    for _ in range(max_passes):
        try:
            return {"json": json.loads(doc), "json_text": doc, "error": ""}
        except JSONDecodeError as e:
            last_error = "{} at pos {} (line {}, col {})".format(e.msg, e.pos, e.lineno, e.colno)
            start, end = _find_value_bounds(doc, e.pos)
            if start == end:
                break
            new_doc = _quote_span(doc, start, end)
            if new_doc == doc:
                break
            doc = new_doc

    return {"json": None, "json_text": doc, "error": "Could not fully parse. Last error: " + last_error}


# if __name__ == "__main__":
#     response_path = "/home/changshu/RE2-Bench-prompt/results/output_prediction/gpt-4o-mini/difficult/scikit-learn__scikit-learn-12682@@sklearn.decomposition.dict_learning.py@@_sparse_encode.txt"
    
#     response_text = open(response_path, "r").read()
    
#     result = extract_output_json_easy(response_text)
    
#     print(json.dumps(result["json"], indent=2) if result["json"] is not None else None)