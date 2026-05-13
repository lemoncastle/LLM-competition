import json
from pathlib import Path

# Load modifications with explicit UTF-8 encoding
mods = {}
with open('./results/modified_answers.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            m = json.loads(line)
            mods[m['index']] = m['answer']

# Read, update, and write batch file
entries = []
with open('./results/batch_output.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            entry = json.loads(line)
            if entry['index'] in mods:
                for msg in entry['messages']:
                    if msg['role'] == 'assistant':
                        msg['content'] = mods[entry['index']]
                        break
            entries.append(entry)

# Write directly without backup with explicit UTF-8 encoding
with open('./results/batch_output.jsonl', 'w', encoding='utf-8') as f:
    for entry in entries:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"Updated {len(mods)} entries")