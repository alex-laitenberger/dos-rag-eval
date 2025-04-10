import os
import json
import csv

def load_jsonl(file_path):
    """Load a JSONL file and return a list of dictionaries."""
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

def evaluate_accuracy(folder_path):
    """Evaluate accuracy and token usage for JSONL files in a given folder."""
    file_list = [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.endswith(".jsonl")
    ]
    
    print("File\tTotal Entries\tCorrect\tAccuracy (%)\tAvg Tokens\tHard Entries\tHard Correct\tHard Accuracy (%)\tNon-Hard Entries\tNon-Hard Correct\tNon-Hard Accuracy (%)")
    
    results = []
    
    for file_path in file_list:
        data = load_jsonl(file_path)
        total = len(data)
        correct = sum(1 for entry in data if entry.get("correct_choice"))
        accuracy = (correct / total) * 100 if total > 0 else 0

        # Calculate average tokens spent
        total_tokens = sum(entry.get("used_tokens", 0) for entry in data)
        avg_tokens = (total_tokens / total) if total > 0 else 0

        # Separate results for hard and non-hard
        hard_data = [entry for entry in data if entry.get("hard", 0) == 1]
        non_hard_data = [entry for entry in data if entry.get("hard", 0) == 0]
        
        hard_total = len(hard_data)
        hard_correct = sum(1 for entry in hard_data if entry.get("correct_choice"))
        hard_accuracy = (hard_correct / hard_total) * 100 if hard_total > 0 else 0

        non_hard_total = len(non_hard_data)
        non_hard_correct = sum(1 for entry in non_hard_data if entry.get("correct_choice"))
        non_hard_accuracy = (non_hard_correct / non_hard_total) * 100 if non_hard_total > 0 else 0

        # Print results with a comma as decimal separator
        print(f"{os.path.basename(file_path)}\t{total}\t{correct}\t{accuracy:.2f}\t{avg_tokens:.2f}\t{hard_total}\t{hard_correct}\t{hard_accuracy:.2f}\t{non_hard_total}\t{non_hard_correct}\t{non_hard_accuracy:.2f}".replace(".", ","))
        
        # Append results for CSV output
        results.append({
            "file": os.path.basename(file_path),
            "total_entries": total,
            "correct": correct,
            "accuracy": round(accuracy, 2),
            "avg_tokens": round(avg_tokens, 2),
            "hard_entries": hard_total,
            "hard_correct": hard_correct,
            "hard_accuracy": round(hard_accuracy, 2),
            "non_hard_entries": non_hard_total,
            "non_hard_correct": non_hard_correct,
            "non_hard_accuracy": round(non_hard_accuracy, 2)
        })
    
    return results

def save_results_csv(results, output_csv):
    """Save evaluation results to a CSV file with ';' as separator for Excel compatibility."""
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["File", "Total Entries", "Correct", "Accuracy (%)", "Avg Tokens", "Hard Entries", "Hard Correct", "Hard Accuracy (%)", "Non-Hard Entries", "Non-Hard Correct", "Non-Hard Accuracy (%)"])
        
        for result in results:
            writer.writerow([
                result["file"],
                result["total_entries"],
                result["correct"],
                str(result["accuracy"]).replace(".", ","),
                str(result["avg_tokens"]).replace(".", ","),
                result["hard_entries"],
                result["hard_correct"],
                str(result["hard_accuracy"]).replace(".", ","),
                result["non_hard_entries"],
                result["non_hard_correct"],
                str(result["non_hard_accuracy"]).replace(".", ",")
            ])
    
    print(f"CSV results saved to {output_csv}")

def evaluate_and_log(folder_path, output_json, output_csv):
    """Evaluate accuracy, token usage, and save results to JSON and CSV."""
    results = evaluate_accuracy(folder_path)

    # Save JSON results
    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"JSON results saved to {output_json}")

    # Save CSV results
    save_results_csv(results, output_csv)

if __name__ == "__main__":
    # Path to the folder with JSONL files
    ANSWERS_FOLDER = "experiments/artifacts/answers/quality/dev"
    
    # Output files
    OUTPUT_JSON = "experiments/artifacts/answers/quality/dev/evaluation_results.json"
    OUTPUT_CSV = "experiments/artifacts/answers/quality/dev/evaluation_results.csv"
    
    # Evaluate and save results
    evaluate_and_log(ANSWERS_FOLDER, OUTPUT_JSON, OUTPUT_CSV)
