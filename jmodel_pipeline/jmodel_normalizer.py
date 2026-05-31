import re

BOX = r"\boxed{"


def extract_last_boxed(text: str):
    start = text.rfind(BOX)
    if start == -1:
        return None

    i = start + len(BOX)
    depth = 1
    while i < len(text) and depth:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1

    if depth != 0:
        return None
    return text[start + len(BOX) : i - 1], start, i


def normalize_function_args_for_judger(s: str) -> str:
    funcs = r"ln|log|sin|cos|tan|sec|csc|cot|exp"
    return re.sub(rf"\\({funcs})\s+([A-Za-z0-9.]+)", r"\\\1(\2)", s)


def normalize_for_judger(s: str) -> str:
    s = s.strip()
    s = normalize_function_args_for_judger(s)

    s = s.replace(r"-\infty", "-infinity").replace("-âˆž", "-infinity")
    s = re.sub(r"(?<![A-Za-z])-inf(?![A-Za-z])", "-infinity", s)
    s = s.replace(r"\infty", "infinity").replace("âˆž", "infinity")
    s = re.sub(r"(?<![A-Za-z])inf(?![A-Za-z])", "infinity", s)

    s = re.sub(r"\\lceil\s*(.*?)\s*\\rceil", r"ceil(\1)", s)
    s = re.sub(r"\\lfloor\s*(.*?)\s*\\rfloor", r"floor(\1)", s)
    s = re.sub(r"e\^\{([^{}]+)\}", r"e^(\1)", s)

    s = re.sub(r"sqrt\(([^()]*)\)", r"\\sqrt{\1}", s)
    s = re.sub(r"(?<!\\)sqrt\{([^{}]+)\}", r"\\sqrt{\1}", s)

    s = re.sub(
        r"(-?)\\frac\{([^{}]*\\sqrt\{[^{}]+\}[^{}]*)\}\{([^{}]+)\}",
        r"\1(\2)/(\3)",
        s,
    )
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/(\2)", s)
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
