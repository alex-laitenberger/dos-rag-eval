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
    
    print("File\tTotal Entries\tCorrect\tAccuracy (%)\tAvg Tokens")
    
    results = []
    
    for file_path in file_list:
        data = load_jsonl(file_path)
        total = len(data)
        correct = sum(1 for entry in data if entry.get("correct_choice"))
        accuracy = (correct / total) * 100 if total > 0 else 0

        # Calculate average tokens spent
        total_tokens = sum(entry.get("used_tokens", 0) for entry in data)
        avg_tokens = (total_tokens / total) if total > 0 else 0

        # Print results with a comma as decimal separator
        print(f"{os.path.basename(file_path)}\t{total}\t{correct}\t{accuracy:.2f}\t{avg_tokens:.2f}"
              .replace(".", ","))
        
        # Append results for CSV output
        results.append({
            "file": os.path.basename(file_path),
            "total_entries": total,
            "correct": correct,
            "accuracy": round(accuracy, 2),
            "avg_tokens": round(avg_tokens, 2)
        })
    
    return results

def save_results_csv(results, output_csv):
    """Save evaluation results to a CSV file with ';' as separator for Excel compatibility."""
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["File", "Total Entries", "Correct", "Accuracy (%)", "Avg Tokens"])
        
        for result in results:
            writer.writerow([
                result["file"],
                result["total_entries"],
                result["correct"],
                str(result["accuracy"]).replace(".", ","),
                str(result["avg_tokens"]).replace(".", ",")
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
    ANSWERS_FOLDER = "experiments/artifacts/answers/infinity_bench/longbook_choice_eng"
    
    # Output files
    OUTPUT_JSON = "experiments/artifacts/answers/infinity_bench/longbook_choice_eng/evaluation_results.json"
    OUTPUT_CSV = "experiments/artifacts/answers/infinity_bench/longbook_choice_eng/evaluation_results.csv"
    
    # Evaluate and save results
    evaluate_and_log(ANSWERS_FOLDER, OUTPUT_JSON, OUTPUT_CSV)
