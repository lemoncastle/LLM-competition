# import sys
# sys.path.insert(0, ".")

# from judger import Judger

# judger = Judger(strict_extract=False)

# expected = ["Yes","(-7-sqrt(41))/2","(-7+sqrt(41))/2",]

# tests = [
#     r"\boxed{Yes, (-7-sqrt(41))/2, (-7+sqrt(41))/2}",
#     r"\boxed{Yes}, \boxed{(-7-\sqrt{41})/2}, \boxed{(-7+\sqrt{41})/2}",
#     r"\boxed{Yes, (-7-sqrt(41))/2, (-7+sqrt(41))/2}",
#     r"\boxed{Yes, (-7-\sqrt{41})/2, (-7+\sqrt{41})/2}"
# ]

# for t in tests:
#     print("-" * 80)
#     print(t)

#     try:
#         result = judger.auto_judge(
#             pred=t,
#             gold=expected,
#             options=[[]] * len(expected),
#         )
#     except Exception as e:
#         result = e

#     print("RESULT:", result)

# wasn't sure why sqrt wasn't working. turns out outputs should be in latex format so they can be extracted properly. the judger will convert latex to what is expected.
# this is confusing since some questions request 'don't output \sqrt, output sqrt instead' but for the judger to work, the output must be in latex format. 
# We also can't change judger.py as it's the same code that is being used to grade our submissions. 

# An interesting quirk is outputing each in its seperate \boxed{} seems to be the best and most reliable way to get a correct answer.

import sys
sys.path.insert(0, ".")

from judger import Judger

judger = Judger(strict_extract=False)

expected = ["sqrt(30)/(3+x)"]

tests = [
    r"\boxed{\frac{\sqrt{30}}{(3+x)}}",
    
]

for t in tests:
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

import re

def normalize_for_frozen_judger(ans: str) -> str:
    s = ans.strip()

    # sqrt(30) -> \sqrt{30}
    s = re.sub(r"sqrt\(([^()]*)\)", r"\\sqrt{\1}", s)
    s = re.sub(r"(?<!\\)sqrt\{([^{}]+)\}", r"\\sqrt{\1}", s)

    # \frac{\sqrt{30}}{3+x} -> \sqrt{30}/(3+x)
    s = re.sub(
        r"\\frac\{(\\sqrt\{[^{}]+\})\}\{([^{}]+)\}",
        r"\1/(\2)",
        s,
    )

    # \frac{a}{b} -> a/(b), for simple non-nested numerator
    s = re.sub(
        r"\\frac\{([^{}]+)\}\{([^{}]+)\}",
        r"\1/(\2)",
        s,
    )

    # Remove redundant denominator parentheses: /((3+x)) -> /(3+x)
    s = re.sub(r"/\(\(([^()]*)\)\)", r"/(\1)", s)

    return s

def boxed_for_judger(ans: str) -> str:
    return r"\boxed{" + normalize_for_frozen_judger(ans) + "}"

print(boxed_for_judger(r"\frac{\sqrt{30}}{(3+x)}"))