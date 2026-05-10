# test deepseek api with a sample question and score it using judger
# also testing prompting to get proper final answer
import os
import sys
import json
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

def generate_question_by_id(question_id: int):
    """Generate response for a specific question ID"""
    parent_dir = Path(__file__).parent.parent
    env_path = parent_dir / '.env'
    load_dotenv(dotenv_path=env_path)

    # Initialize the client
    client = OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )

    # Load the dataset and find the question by ID
    public_data = [json.loads(line) for line in open("./data/public.jsonl")]
    item = next((x for x in public_data if x["id"] == question_id), None)
    
    if item is None:
        print(f"Question with ID {question_id} not found!")
        return None
    
    question = item["question"]
    options = item.get("options")
    
    # Choose prompt based on question type
    if options:
        system_prompt = (
            "You are an expert mathematician. Solve the problem step-by-step. "
            "Use the answer choices to determine the correct option. "
            "Put your final answer inside \\boxed{<letter>}. "
        )
        # Format options for display
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        user_content = f"{question}\n\nOptions:\n{opts_text}"
    else:
        system_prompt = (
            "You are an expert mathematician. Solve the problem step-by-step. "
            "Put your final answer inside \\boxed{}. "
            "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}. "
            "If a part has multiple values, group those values in parentheses. "
        )
        user_content = question

    s = datetime.now()
    print(f"Generating response for id {question_id}...")
    
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        extra_body={"thinking": {"type": "enabled"}},
        max_tokens=16384,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
    )
    
    print("Response generated in", datetime.now() - s)
    print("Tokens used:", response.usage.total_tokens)

    # Get the thinking content and final answer
    thinking_content = getattr(response.choices[0].message, 'reasoning_content', None)
    final_answer = response.choices[0].message.content
    full_response = f"<think>\n{thinking_content}\n</think>\n\n{final_answer}"

    # Score it
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
    
    return full_response, correct

def main():
    # You can change this to any ID you want
    question_id = 159
    
    result = generate_question_by_id(question_id)
    print(result[0])
    print(result[1])

if __name__ == "__main__":
    main()