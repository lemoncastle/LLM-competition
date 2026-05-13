import json
from pathlib import Path

# Load modifications with explicit UTF-8 encoding

mod_path = './results/modified_answers.jsonl'
batch_path = './results/batch_output_roll.jsonl'

mods = {}
with open(mod_path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            m = json.loads(line)
            mods[m['index']] = m['answer']

# Read, update, and write batch file
entries = []
with open(batch_path, 'r', encoding='utf-8') as f:
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
with open(batch_path, 'w', encoding='utf-8') as f:
    for entry in entries:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"Updated {len(mods)} entries")