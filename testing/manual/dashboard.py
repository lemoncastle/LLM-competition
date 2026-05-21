import json
import webbrowser
from pathlib import Path
from datetime import datetime

results_path = Path('./results/evaluation_results.jsonl')
modified_path = Path('./results/modified_answers.jsonl')

# Load your incorrect results (for review)
with open(results_path, 'r', encoding='utf-8') as f:
    incorrect_items = [json.loads(line) for line in f]

# Prepare items data for review
# Load already modified IDs (if file exists)
modified_ids = set()
if modified_path.exists():
    with open(modified_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line)
                modified_ids.add(entry.get('index'))
            except:
                pass

# Prepare items data for review
items_data = []
count_skipped = 0
for idx, item in enumerate(incorrect_items):
    item_id = item.get('id', idx)

    # Skip already modified items
    if item_id in modified_ids:
        count_skipped += 1
        continue

    expected = item.get('gold', item.get('correct_answer', 'N/A'))
    if isinstance(expected, list):
        expected = ', '.join(expected)
    
    items_data.append({
        'id': item_id,
        'question': (
            item.get('question', 'N/A') +
            (
                '<br>Options:<br>' +
                ', '.join(
                    f"{chr(65 + i)}. {opt}"
                    for i, opt in enumerate(item.get('options', []))
                )
                if item.get('options')
                else ''
            )
        ),
        'generated_answer': item.get('response', 'N/A'),
        'expected_answer': expected,
        'is_mcq': item.get('is_mcq', False),
        'correct': item.get('correct', False),
        'original_response': item.get('response', 'N/A'),
        'index': idx
    })

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
        
        // Initialize modified_answer for each item
        for (let i = 0; i < items.length; i++) {
            if (!items[i].modified_answer) {
                items[i].modified_answer = items[i].generated_answer;
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
            const modified = items.filter(i => i.saved).length;
            
            const div = document.getElementById('items');
            div.innerHTML = `
                <div class="item">
                    <div class="header">
                        <span>
                            <strong>Item ID: ${escapeHtml(String(item.id))}</strong> |
                            Type: ${item.is_mcq ? 'MCQ' : 'FRQ'} |
                            Correct: ${item.correct ? 'True' : 'False'}
                        </span>
                        <span class="nav-info">Item ${currentIndex + 1} of ${items.length} | Modified: ${modified}/${items.length}</span>
                    </div>
                    <div class="question">
                        <p>Question:</p>
                        <div>${String(item.question)}</div>
                    </div>
                    <div class="expected">
                        <p>Expected (correct answer):</p>
                        <div>${escapeHtml(String(item.expected_answer))}</div>
                    </div>
                    <div class="actions">
                        <div class="fix-label">Modified Answer (edit if needed):</div>
                        <textarea id="modifiedAnswer" placeholder="Enter corrected answer...">${escapeHtml(String(item.modified_answer || item.generated_answer))}</textarea>
                        <br>
                        <div style="display:flex; align-items:center; gap:8px;">
                            <button class="btn-format" onclick="saveAndNext()">✓ Save & Next</button>
                            <button class="btn-skip" onclick="skipAndNext()">⏭ Skip (keep original)</button>
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
            const modifiedAnswer = document.getElementById('modifiedAnswer').value;
            items[currentIndex].modified_answer = modifiedAnswer;
            items[currentIndex].saved = true;
            
            const statusDiv = document.getElementById('status');
            statusDiv.textContent = '✓ Modified answer saved';
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
            items[currentIndex].modified_answer = items[currentIndex].original_response;
            
            const statusDiv = document.getElementById('status');
            statusDiv.textContent = '⏭ Skipped (keeping original)';
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
                const modifiedAnswer = document.getElementById('modifiedAnswer');
                if (modifiedAnswer) {
                    items[currentIndex].modified_answer = modifiedAnswer.value;
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
            const modifiedAnswer = document.getElementById('modifiedAnswer');
            if (modifiedAnswer && !items[currentIndex].modified_answer) {
                items[currentIndex].modified_answer = modifiedAnswer.value;
            }
            
            // Save answers regardless of modification
            const modifiedItems = items.filter(function(item) {
                return item.saved;
            });
            
            // Create output in jsonl format
            let jsonlContent = '';
            for (let i = 0; i < modifiedItems.length; i++) {
                const item = modifiedItems[i];
                const outputEntry = {
                    index: item.id,
                    question: item.question,
                    answer: item.modified_answer
                };
                jsonlContent += JSON.stringify(outputEntry) + '\\n';
            }
            
            // Create download
            const blob = new Blob([jsonlContent], {type: 'application/x-jsonlines'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'modified_answers.jsonl';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            const totalModified = modifiedItems.length;
            const itemsDiv = document.getElementById('items');
            if (itemsDiv) {
                itemsDiv.innerHTML = `
                    <div style="text-align: center; padding: 40px;">
                        <h2>Modified Answers Saved!</h2>
                        <p>Downloaded: <strong>modified_answers.jsonl</strong></p>
                        <br>
                        <h3>Summary:</h3>
                        <ul style="text-align: left; display: inline-block; background: #faf8f5; padding: 20px 40px; border-radius: 12px;">
                            <li>Total items reviewed: ${items.length}</li>
                            <li>Modified answers saved: ${totalModified}</li>
                            <li>⏭ Skipped/unmodified: ${items.length - totalModified}</li>
                        </ul>
                        <br><br>
                        <p style="color: #8b7a6b;">The file <strong>modified_answers.jsonl</strong> contains only the answers you changed.</p>
                        <p style="color: #8b7a6b;">Format: {"index": N, "question": "...", "answer": "..."} per line</p>
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

print(f"{len(incorrect_items)} items loaded from {results_path}")
print(f"{len(items_data)} items left to review with {count_skipped} already reviewed.")