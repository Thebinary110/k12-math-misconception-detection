"""Parse VLM transcriptions and compare a student's final answer with the answer key."""
import re
from fractions import Fraction


def strip_notes(text):
    """Remove annotator notes ([...]) from gold transcriptions; they are absent at test time and can leak the label."""
    return re.sub(r"\[[^\]]*\]", "", str(text)).strip()


def lines_and_final(text):
    lines, final = [], ""
    for raw in str(text).splitlines():
        m = re.match(r"\s*(L\d+|FINAL)\s*\|\s*(.*)", raw)
        if not m:
            continue
        if m.group(1) == "FINAL":
            final = m.group(2).strip()
        else:
            lines.append(m.group(2).strip())
    if not final and lines:
        final = lines[-1].split("=")[-1].strip()
    return lines, ("" if final.lower() in ("none", "-", "") else final)


def norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\sqrt\{([^{}]*)\}", r"√(\1)", s)
    s = re.sub(r"\\(left|right|text|mathrm|displaystyle|,|;|!|quad)", "", s)
    s = (s.replace("\\times", "×").replace("\\cdot", "×").replace("*", "×").replace("x", "x")
         .replace("\\div", "÷").replace("−", "-").replace("–", "-").replace("^2", "²").replace("^3", "³"))
    s = re.sub(r"[\s$\{\}\\]", "", s)
    s = re.sub(r"^(jadi|jawab(an)?|maka|hasil(nya)?|=)+", "", s)
    s = re.sub(r"^[a-z]=", "", s)
    s = re.sub(r"(?<=\d)\.(?=\d{3}\b)", "", s)  # Indonesian thousands separator 1.000
    s = s.replace(",", ".").rstrip(".")
    return s


def as_number(s):
    try:
        if "/" in s and s.count("/") == 1:
            a, b = s.replace("(", "").replace(")", "").split("/")
            return float(Fraction(a) / Fraction(b))
        return float(s)
    except (ValueError, ZeroDivisionError):
        return None


def answer_matches(student, key):
    a, b = norm(student), norm(key)
    if not a:
        return False
    if a == b:
        return True
    x, y = as_number(a), as_number(b)
    return x is not None and y is not None and abs(x - y) < 1e-6 * max(1, abs(y))
