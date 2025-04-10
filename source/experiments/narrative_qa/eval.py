import argparse
import collections
import json
import os
import re
import string
import sys
import csv
import glob
import math
import numpy as np  # Import numpy for std computation

import evaluate

import locale

try:
    locale.setlocale(locale.LC_NUMERIC, "de_DE.UTF-8")  # Common German locale format
except locale.Error:
    locale.setlocale(locale.LC_NUMERIC, "")  # Fallback to system default

def format_number(value):
    """Formats numbers with European-style separators (comma for decimals, dot for thousands)."""
    if isinstance(value, float):
        return locale.format_string("%.6f", value, grouping=False).replace(".", ",")  # Force decimal comma
    return value  # Leave non-numeric values unchanged


folder_path = "experiments/artifacts/answers/narrative_qa/test"
output_csv = f"{folder_path}/result.csv"
output_json = f"{folder_path}/result.json"

bleu = evaluate.load("bleu")
squad_metric = evaluate.load("squad_v2")
rouge = evaluate.load('rouge')
meteor = evaluate.load('meteor')

def is_valid_string(value):
    """Check if value is a valid string (not None, NaN, or a non-string type)."""
    return isinstance(value, str) and value.strip() != ""

def process_file(file_path):
    """Processes a JSONL file and computes F1 scores."""
    num_documents = set()
    num_questions = 0
    total_used_tokens = 0
    f1_scores = {}
    bleu1_scores = {}
    bleu4_scores = {}
    predictions = []
    references = []
    squad_predictions = []
    squad_references = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)

            document_id = entry["document_id"]
            question_id = entry["question_id"]
            gold_answers = entry["gold_answers"]
            predicted_answer = entry["predicted_answer"]
            used_tokens = entry.get("used_tokens", 0)  # Default to 0 if missing

            # Ensure the predicted answer is a valid string
            if not is_valid_string(predicted_answer):
                continue  # Skip if prediction is not a valid string

            # Filter out non-string elements from gold_answers
            valid_references = [a for a in gold_answers if is_valid_string(a)]

            if valid_references: 
                num_documents.add(document_id)
                num_questions += 1
                total_used_tokens += used_tokens
                predictions.append(predicted_answer)
                squad_predictions.append({"id": question_id, "prediction_text": predicted_answer, "no_answer_probability": 0.0})
                references.append(valid_references)
                squad_references.append({"id": question_id, "answers": [{"text": ans, "answer_start": 0} for ans in valid_references]})

    meteor_results = meteor.compute(predictions=predictions, references=references)
    rouge_results = rouge.compute(predictions=predictions, references=references)
    squad_results = squad_metric.compute(predictions=squad_predictions, references=squad_references)            
    bleu_1_results = bleu.compute(predictions=predictions, references=references, max_order=1, smooth = False) #smooth = False is default
    bleu_4_results = bleu.compute(predictions=predictions, references=references, max_order=4, smooth = False) #smooth = False is default
    avg_used_tokens = total_used_tokens / num_questions if num_questions > 0 else 0

    return {
        "filename": os.path.basename(file_path),
        "num_documents": len(num_documents),
        "num_questions": num_questions,
        "average_used_tokens": format_number(avg_used_tokens),
        "f1": format_number(squad_results['f1']),
        "squad_total": format_number(squad_results['total']),
        "bleu1_score": format_number(bleu_1_results['bleu']), 
        "bleu4_score": format_number(bleu_4_results['bleu']),
        "rouge_L": format_number(rouge_results['rougeL']),
        "meteor": format_number(meteor_results['meteor'])

    }


def process_folder():
    """Processes all JSONL files in a folder and writes results to CSV and JSON."""
    results = []
    json_results = []

    for file_path in glob.glob(os.path.join(folder_path, "*.jsonl")):
        file_result = process_file(file_path)
        results.append([
            file_result["filename"],
            file_result["num_documents"],
            file_result["num_questions"],
            file_result["average_used_tokens"],
            file_result["f1"],
            file_result["squad_total"],
            file_result["bleu1_score"],
            file_result["bleu4_score"],
            file_result["rouge_L"],
            file_result["meteor"]
        ])
        json_results.append(file_result)

    # Write CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file, delimiter=";")
        writer.writerow(["filename", "num_documents", "num_questions", "average_used_tokens", "f1", "squad_total", "bleu-1", "bleu-4", "rouge-L", "meteor" ])
        writer.writerows(results)

    # Write JSON
    with open(output_json, "w", encoding="utf-8") as json_file:
        json.dump(json_results, json_file, indent=4)


if __name__ == "__main__":
    process_folder()