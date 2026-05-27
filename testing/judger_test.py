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

expected = ["infinity"]

tests = [
    r"\boxed{\infty}",
    
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