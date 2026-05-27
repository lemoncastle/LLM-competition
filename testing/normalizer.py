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


def normalize_for_judger(s: str) -> str:
    s = s.strip()

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

# import json

# INPUT_JSONL = "./results/sft_train_distilled.jsonl"
# OUTPUT_JSONL = "normalized.jsonl"

# with open(INPUT_JSONL, "r", encoding="utf-8") as fin, \
#      open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:

#     for line in fin:
#         line = line.strip()

#         if not line:
#             continue

#         obj = json.loads(line)

#         # normalize assistant messages only
#         for msg in obj.get("messages", []):
#             if msg.get("role") == "assistant":
#                 msg["content"] = normalize_assistant_output(
#                     msg["content"]
#                 )

#         fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

# print("done")

# I'm kinda stupid and saved inference data in full.py as csv instead of jsonl.
import pandas as pd

INPUT_CSV = "input.csv"
OUTPUT_CSV = "normalized.csv"

df = pd.read_csv(INPUT_CSV)

# normalize assistant responses
df["response"] = df["response"].apply(normalize_assistant_output)

df.to_csv(OUTPUT_CSV, index=False)

print("done")