import re
import sys
sys.path.insert(0, ".")

from judger import Judger

judger = Judger(strict_extract=False)

import re

BOX = r"\boxed{"

def extract_last_boxed(text: str) -> tuple[str, int, int] | None:
    start = text.rfind(BOX)
    if start == -1:
        return None

    i = start + len(BOX)
    depth = 1

    while i < len(text) and depth:
        depth += text[i] == "{"
        depth -= text[i] == "}"
        i += 1

    if depth:
        return None

    return text[start + len(BOX): i - 1], start, i

def normalize_function_args_for_judger(s: str) -> str:
    # \ln 10 -> \ln(10)
    # \log 10 -> \log(10)
    # \sin x -> \sin(x), etc.
    funcs = r"ln|log|sin|cos|tan|sec|csc|cot|exp"
    s = re.sub(
        rf"\\({funcs})\s+([A-Za-z0-9.]+)",
        r"\\\1(\2)",
        s,
    )
    return s

def normalize_for_judger(s: str) -> str:
    s = s.strip()
    s = normalize_function_args_for_judger(s)
    
    # infinity
    s = s.replace(r"-\infty", "-infinity").replace("-∞", "-infinity")
    s = re.sub(r"(?<![A-Za-z])-inf(?![A-Za-z])", "-infinity", s)
    s = s.replace(r"\infty", "infinity").replace("∞", "infinity")
    s = re.sub(r"(?<![A-Za-z])inf(?![A-Za-z])", "infinity", s)

    # ceil / floor
    s = re.sub(r"\\lceil\s*(.*?)\s*\\rceil", r"ceil(\1)", s)
    s = re.sub(r"\\lfloor\s*(.*?)\s*\\rfloor", r"floor(\1)", s)

    # e^{16x} -> e^(16x)
    s = re.sub(r"e\^\{([^{}]+)\}", r"e^(\1)", s)

    # sqrt(30), sqrt{30} -> \sqrt{30}
    s = re.sub(r"sqrt\(([^()]*)\)", r"\\sqrt{\1}", s)
    s = re.sub(r"(?<!\\)sqrt\{([^{}]+)\}", r"\\sqrt{\1}", s)

    # \frac{anything with \sqrt{...}}{b} -> (anything)/(b)
    s = re.sub(
        r"(-?)\\frac\{([^{}]*\\sqrt\{[^{}]+\}[^{}]*)\}\{([^{}]+)\}",
        r"\1(\2)/(\3)",
        s,
    )
    # complex frac -> slash form
    s = re.sub(
        r"\\frac\{((?:[^{}]|\{[^{}]*\})+)\}\{((?:[^{}]|\{[^{}]*\})+)\}",
        r"(\1)/(\2)",
        s,
    )

    # simple \frac{a}{b} -> a/(b)
    s = re.sub(
        r"\\frac\{([^{}]+)\}\{([^{}]+)\}",
        r"\1/(\2)",
        s,
    )

    # cleanup
    s = re.sub(r"/\(\(([^()]*)\)\)", r"/(\1)", s)

    return s


def normalize_assistant_output(content: str) -> str:
    found = extract_last_boxed(content)

    if found is None:
        if "</think>" in content:
            prefix, answer = content.rsplit("</think>", 1)
            return prefix + "</think>\n\n" + BOX + normalize_for_judger(answer) + "}"
        return BOX + normalize_for_judger(content) + "}"

    boxed, start, end = found
    return content[:start] + BOX + normalize_for_judger(boxed) + "}" + content[end:]

content = r"""<think>
Some thinking response.
</think>

\boxed{\sin(\alpha)=\frac{5}{\sqrt{29}},\ \cos(\alpha)=-\frac{2}{\sqrt{29}},\ \cot(\alpha)=-2/(5),\ \sec(\alpha)=-(\sqrt{29})/(2),\ \csc(\alpha)=(\sqrt{29})/(5)}"""

print("ORIGINAL:", content)
norm = normalize_assistant_output(content)
print("NORMALIZED:", norm)

expected = ["-sqrt(13)/4","0", "sqrt(13)/4", "0", "0", "-13", "-13", "+INF"]

t = norm

print("-" * 80)
print("RAW:", t)

extracted = judger.extract_ans(t)
split_pred = judger.split_by_comma(extracted)

norm_pred = [judger.norm_ans_str(x) for x in split_pred]
norm_gold = [judger.norm_ans_str(x) for x in expected]

print("NORM PRED:", norm_pred)
print("NORM GOLD:", norm_gold)

result = judger.auto_judge(
    pred=t,
    gold=expected,
    options=[[]] * len(expected),
)

print(" == RESULT == ", result)

for p, g in zip(norm_pred, norm_gold):
    print("PAIR:", repr(p), "vs", repr(g), "=>", judger.is_equal(p, g))