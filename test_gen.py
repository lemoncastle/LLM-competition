import json

# Sample result data
results = [json.loads(line) for line in open("./results/fo.jsonl").readlines()]

def calculate_percentage(results):
    mcq_correct = 0
    mcq_total = 0
    frq_correct = 0
    frq_total = 0
    
    for result in results:
        if result['is_mcq']:
            mcq_total += 1
            if result['correct']:
                mcq_correct += 1
        else:
            frq_total += 1
            if result['correct']:
                frq_correct += 1

    # Calculate percentages
    mcq_percentage = (mcq_correct / mcq_total) * 100 if mcq_total > 0 else 0
    frq_percentage = (frq_correct / frq_total) * 100 if frq_total > 0 else 0

    return mcq_percentage, frq_percentage, mcq_correct, mcq_total, frq_correct, frq_total


# Call the function to calculate the percentages
mcq_percentage, frq_percentage, mcq_correct, mcq_total, frq_correct, frq_total = calculate_percentage(results)

# Output the results
print(f"MCQ - Correct: {mcq_correct}/{mcq_total} = {mcq_percentage:.2f}%")
print(f"FRQ - Correct: {frq_correct}/{frq_total} = {frq_percentage:.2f}%")
print(f"Overall - Correct: {mcq_correct + frq_correct}/{mcq_total + frq_total} = {(mcq_correct + frq_correct) / (mcq_total + frq_total) * 100:.2f}%")