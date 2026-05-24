import csv
import json
import webbrowser
from pathlib import Path

# Inputs expected in the working directory:
#   submission.csv  with at least columns: id,response
#   private.jsonl   with question records that include matching id values
submission_path = Path('./results/submission.csv')
private_path = Path('./data/private.jsonl')
hint_path = Path('./results/prompt_hints.jsonl')


def normalize_id(value):
    """Normalize ids so CSV ids like 1 and JSON ids like "1" match."""
    if value is None:
        return ''
    text = str(value).strip()
    # Treat 1.0 as 1, but leave non-numeric IDs alone.
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def load_jsonl(path):
    """Load a JSONL file, ignoring blank lines and reporting malformed lines."""
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"Skipping malformed JSON on line {line_number}: {exc}")
    return items


def load_submission_csv(path):
    """Load submission.csv and return rows with normalized `id` and `response` fields.

    Handles normal comma CSV, tab-separated pasted exports, and files with an extra
    unnamed index column before id/response.
    """
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=',\t')
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        rows = []
        for row_number, row in enumerate(reader, start=2):
            # Drop empty/unnamed index columns such as "" or None.
            clean = {str(k).strip(): v for k, v in row.items() if k is not None and str(k).strip()}
            lower_to_key = {k.lower(): k for k in clean}

            id_key = lower_to_key.get('id')
            response_key = (
                lower_to_key.get('response')
                or lower_to_key.get('answer')
                or lower_to_key.get('prediction')
                or lower_to_key.get('output')
            )

            if not id_key or not response_key:
                print(f"Skipping CSV row {row_number}: could not find id/response columns")
                continue

            rows.append({
                'id': normalize_id(clean.get(id_key)),
                'response': clean.get(response_key, '') or '',
                'row_number': row_number,
            })
        return rows


def get_question_text(item):
    """Find question text in common private.jsonl schemas."""
    for key in ('question', 'prompt', 'problem', 'query', 'input'):
        if item.get(key):
            return str(item[key])

    # Chat-style fallback: first user message.
    for message in item.get('messages', []):
        if message.get('role') == 'user':
            return str(message.get('content', ''))

    return 'N/A'


def format_options(options):
    """Format options as HTML. Accepts list, dict, or missing options."""
    if not options:
        return 'N/A'
    if isinstance(options, dict):
        return '<br>'.join(
            f"{escape_for_html(str(label))}. {escape_for_html(str(text))}"
            for label, text in options.items()
        )
    if isinstance(options, list):
        return '<br>'.join(
            f"{chr(65 + i)}. {escape_for_html(str(option))}"
            for i, option in enumerate(options)
        )
    return escape_for_html(str(options))


def escape_for_html(text):
    """Minimal escaping for question/options inserted as HTML."""
    return (
        text.replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&#39;')
            .replace('\n', '<br>')
    )


def private_item_id(item, fallback_idx):
    """Find the ID field in common private.jsonl schemas."""
    for key in ('id', 'index', 'qid', 'question_id'):
        if key in item:
            return normalize_id(item.get(key))
    return normalize_id(fallback_idx)


def submission_to_review_row(sub_row, private_by_id, fallback_idx):
    item_id = normalize_id(sub_row.get('id'))
    private_item = private_by_id.get(item_id, {})

    question = escape_for_html(get_question_text(private_item)) if private_item else 'N/A - no matching private.jsonl row found'
    options = private_item.get('options', private_item.get('choices', private_item.get('answers', [])))
    expected = private_item.get('gold', private_item.get('correct_answer', private_item.get('answer', 'N/A')))
    if isinstance(expected, list):
        expected = ', '.join(map(str, expected))

    return {
        'id': item_id,
        'question': question,
        'generated_answer': sub_row.get('response', ''),
        'expected_answer': expected,
        'options_html': format_options(options),
        'is_mcq': bool(options),
        'correct': False,
        'matched_private': bool(private_item),
        'original_response': sub_row.get('response', ''),
        'prompt_hint': '',
        'index': fallback_idx,
    }


submission_items = load_submission_csv(submission_path)
private_items = load_jsonl(private_path)
private_items_by_id = {
    private_item_id(item, idx): item
    for idx, item in enumerate(private_items)
}

# Load already reviewed IDs (if file exists)
reviewed_ids = set()
if hint_path.exists():
    with open(hint_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line)
                reviewed_ids.add(normalize_id(entry.get('index')))
            except json.JSONDecodeError:
                pass

items_data = []
count_skipped = 0
for idx, sub_row in enumerate(submission_items):
    item_id = normalize_id(sub_row.get('id'))
    if item_id in reviewed_ids:
        count_skipped += 1
        continue
    items_data.append(submission_to_review_row(sub_row, private_items_by_id, idx))

# Generate review dashboard
html = """
<html>
<head>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f4f6f9; }
        .item { 
            border: 1px solid #e0d5c1; 
            margin: 15px 0; 
            padding: 12px;
            background: #fffef7;
            border-radius: 12px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        }
        .header { 
            background: #b8d4e3;
            color: #3a5a6e; 
            padding: 8px 12px; 
            margin: -12px -12px 12px -12px;
            border-radius: 12px 12px 0 0;
            font-size: 13px;
            font-weight: 600;
        }
        .question { 
            background: #d4eaf7;
            padding: 10px 10px 10px 20px;
            margin: 6px 0; 
            border-left: 4px solid #7fb4cf;
            font-size: 14px;
            border-radius: 0 8px 8px 0;
        }
        .question p {
            margin: 0 0 2px 0;
            font-weight: bold;
            font-size: 14px;
            color: #3a6b8c;
        }
        .expected { 
            background: #d9f0e1;
            padding: 10px 10px 10px 20px;
            margin: 6px 0; 
            border-left: 4px solid #86b49b;
            font-size: 14px;
            border-radius: 0 8px 8px 0;
        }
        .expected p {
            margin: 0 0 2px 0;
            font-weight: bold;
            font-size: 14px;
            color: #2d6a4f;
        }
        .actions { 
            margin: 8px 0; 
            padding-top: 8px;
            border-top: 1px solid #f0e5d8;
        }
        button { 
            margin: 4px; 
            padding: 6px 12px;
            cursor: pointer;
            border: none;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        button:hover { 
            opacity: 0.85;
            transform: translateY(-1px);
        }
        .btn-format { background-color: #b8d9c9; color: #2d5a41; }
        .btn-skip { background-color: #e8e0d8; color: #6b5b4f; }
        .btn-previous { background-color: #c9dde8; color: #3a6b8c; }
        .btn-save-all { background-color: #d4c4e8; color: #4a3a6e; }
        .answer-box {
            color: #4f4f4f;
        }
        .hint-box {
            height: 160px;
            background: #fff;
        }
        textarea { 
            width: 100%; 
            height: 600px; 
            margin: 6px 0;
            padding: 10px;
            border: 2px solid #d4c4e8;
            border-radius: 8px;
            font-size: 14px;
            resize: vertical;
            background: #F5F5F5;
            
        }
        .status { 
            display: inline-block; 
            margin-left: 8px; 
            font-weight: bold; 
            padding: 4px 8px;
            border-radius: 20px;
            font-size: 12px;
        }
        .status-success { background-color: #d4f0e0; color: #2d6a4f; }
        .fix-label {
            font-weight: bold;
            margin: 6px 0 3px 0;
            color: #8b7a6b;
            font-size: 12px;
        }
        .nav-info {
            font-size: 12px;
            font-weight: normal;
            background: rgba(255,255,255,0.3);
            padding: 4px 8px;
            border-radius: 20px;
        }
    </style>
</head>
<body>
    <div id="items"></div>
    
    <script>
        const items = ITEMS_PLACEHOLDER;
        let currentIndex = 0;
        
        // Initialize prompt hint for each item
        for (let i = 0; i < items.length; i++) {
            if (items[i].prompt_hint === undefined || items[i].prompt_hint === null) {
                items[i].prompt_hint = '';
            }
            items[i].saved = false;
        }
        
        function escapeHtml(text) {
            if (!text) return '';
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        function renderItem() {
            const item = items[currentIndex];
            const reviewed = items.filter(i => i.saved).length;
            
            const div = document.getElementById('items');
            div.innerHTML = `
                <div class="item">
                    <div class="header">
                        <span>
                            <strong>Item ID: ${escapeHtml(String(item.id))}</strong> |
                            Type: ${item.is_mcq ? 'MCQ' : 'FRQ'} |
                            Matched private.jsonl: ${item.matched_private ? 'Yes' : 'No'}
                        </span>
                        <span class="nav-info">Item ${currentIndex + 1} of ${items.length} | Reviewed: ${reviewed}/${items.length}</span>
                    </div>
                    <div class="question">
                        <p>Question:</p>
                        <div>${String(item.question)}</div>
                    </div>
                    <div class="expected">
                        <p>Options:</p>
                        <div>${String(item.options_html || escapeHtml(String(item.expected_answer)))}</div>
                    </div>
                    <div class="actions">
                        <div class="fix-label">Submitted Answer (read-only):</div>
                        <textarea id="generatedAnswer" class="answer-box" readonly>${escapeHtml(String(item.generated_answer || ''))}</textarea>
                        <div class="fix-label">Prompt Hint (optional):</div>
                        <textarea id="promptHint" class="hint-box" placeholder="Add a prompt hint for this question if needed...">${escapeHtml(String(item.prompt_hint || ''))}</textarea>
                        <br>
                        <div style="display:flex; align-items:center; gap:8px;">
                            <button class="btn-format" onclick="saveAndNext()">✓ Save & Next</button>
                            <button class="btn-skip" onclick="skipAndNext()">⏭ Skip (save empty hint)</button>
                            <button class="btn-previous" onclick="goToPrevious()">◀ Previous</button>

                            <button class="btn-save-all" onclick="saveAndExit()" style="margin-left:auto;">
                                Save & Exit
                            </button>

                            <span id="status" class="status"></span>
                        </div>
                    </div>
                </div>
            `;
        }
        
        function saveAndNext() {
            const promptHint = document.getElementById('promptHint').value;
            items[currentIndex].prompt_hint = promptHint;
            items[currentIndex].saved = true;
            
            const statusDiv = document.getElementById('status');
            statusDiv.textContent = '✓ Prompt hint saved';
            statusDiv.className = 'status status-success';
            
            setTimeout(() => {
                if (currentIndex + 1 < items.length) {
                    currentIndex++;
                    renderItem();
                } else {
                    saveAndExit();
                }
            }, 300);
        }
        
        function skipAndNext() {
            items[currentIndex].prompt_hint = '';
            items[currentIndex].saved = true;
            
            const statusDiv = document.getElementById('status');
            statusDiv.textContent = '⏭ Skipped (empty hint saved)';
            statusDiv.className = 'status status-success';
            
            setTimeout(() => {
                if (currentIndex + 1 < items.length) {
                    currentIndex++;
                    renderItem();
                } else {
                    saveAndExit();
                }
            }, 300);
        }
        
        function goToPrevious() {
            if (currentIndex > 0) {
                const promptHint = document.getElementById('promptHint');
                if (promptHint) {
                    items[currentIndex].prompt_hint = promptHint.value;
                }
                currentIndex--;
                renderItem();
            } else {
                const statusDiv = document.getElementById('status');
                statusDiv.textContent = 'Already at first item';
                statusDiv.className = 'status status-success';
                setTimeout(() => {
                    if (statusDiv) statusDiv.textContent = '';
                }, 2000);
            }
        }
        
        function saveAndExit() {
            // Save current item if not saved
            const promptHint = document.getElementById('promptHint');
            if (promptHint) {
                items[currentIndex].prompt_hint = promptHint.value;
                items[currentIndex].saved = true;
            }
            
            // Save reviewed items. Empty hints are saved as an empty string.
            const reviewedItems = items.filter(function(item) {
                return item.saved;
            });
            
            // Create output in jsonl format
            let jsonlContent = '';
            for (let i = 0; i < reviewedItems.length; i++) {
                const item = reviewedItems[i];
                const outputEntry = {
                    index: item.id,
                    hint: item.prompt_hint || ''
                };
                jsonlContent += JSON.stringify(outputEntry) + '\\n';
            }
            
            // Create download
            const blob = new Blob([jsonlContent], {type: 'application/x-jsonlines'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'prompt_hints.jsonl';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            const totalReviewed = reviewedItems.length;
            const itemsDiv = document.getElementById('items');
            if (itemsDiv) {
                itemsDiv.innerHTML = `
                    <div style="text-align: center; padding: 40px;">
                        <h2>Prompt Hints Saved!</h2>
                        <p>Downloaded: <strong>prompt_hints.jsonl</strong></p>
                        <br>
                        <h3>Summary:</h3>
                        <ul style="text-align: left; display: inline-block; background: #faf8f5; padding: 20px 40px; border-radius: 12px;">
                            <li>Total items reviewed: ${items.length}</li>
                            <li>Prompt hints saved: ${totalReviewed}</li>
                            <li>Not reviewed this session: ${items.length - totalReviewed}</li>
                        </ul>
                        <br><br>
                        <p style="color: #8b7a6b;">The file <strong>prompt_hints.jsonl</strong> contains the reviewed question IDs and prompt hints.</p>
                        <p style="color: #8b7a6b;">Format: {"index": N, "hint": "..."} per line</p>
                    </div>
                `;
            }
        }
        
        renderItem();
    </script>
</body>
</html>
"""

# Replace the placeholder with actual data
items_json = json.dumps(items_data, ensure_ascii=False)
html = html.replace("ITEMS_PLACEHOLDER", items_json)

# Save and open dashboard
output_path = Path('./review_dashboard.html')
output_path.write_text(html, encoding='utf-8')
webbrowser.open(output_path.absolute().as_uri())

print(f"{len(submission_items)} rows loaded from {submission_path}")
print(f"{len(private_items_by_id)} questions loaded from {private_path}")
print(f"{len(items_data)} items left to review with {count_skipped} already reviewed.")