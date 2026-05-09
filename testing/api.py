# test deepseek api with a sample question and score it using judger
# also testing prompting to get proper final answer
import os
import sys
import json
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

def main():
    parent_dir = Path(__file__).parent.parent
    env_path = parent_dir / '.env'
    load_dotenv(dotenv_path=env_path)

    # Initialize the client
    client = OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )

    # Make the API call with thinking enabled
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        extra_body={"thinking": {"type": "enabled"}},
        messages=[
            {
            "role": "system",
            "content": (
                "You are an expert mathematician. Solve the problem step-by-step. "
                "Put your final answer inside \\boxed{}. "
                "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
                "If a part has multiple values, group those values in parentheses. "
            ),
            },
            
            {"role": "user", "content": "List all values of $c$ that makes each trinomial a perfect square trinomial. Separate multiple answers by commas.\n(a) $x^2+8x+c$: [ANS]\n(b) $x^2+c x+25$: [ANS]"}
        ]
    )

    # Get the thinking content and final answer
    thinking_content = getattr(response.choices[0].message, 'reasoning_content', None)
    final_answer = response.choices[0].message.content

    # Format 1: Combine them with <think> tags (matching DeepSeek's web interface style)
    if thinking_content:
        full_response = f"<think>\n{thinking_content}\n</think>\n\n{final_answer}"
    else:
        full_response = final_answer


    public_data = [json.loads(line) for line in open("./data/public.jsonl")]

    item = next(x for x in public_data if x["id"] == 1120)

    # score it
    sys.path.insert(0, ".")
    from judger import Judger

    judger = Judger(strict_extract=False)

    gold = item["answer"]
    gold_list = gold if isinstance(gold, list) else [gold]

    correct = judger.auto_judge(
        pred=full_response,
        gold=gold_list,
        options=[[]] * len(gold_list),
    )

    print(full_response)
    print("Correct:", correct)

if __name__ == "__main__":
    main()