import pandas as pd
import json
import os
import hashlib

NARRATIVE_QA_PATH = '~/narrativeqa'
INPUT_FILE = f'{NARRATIVE_QA_PATH}/qaps.csv'
DATASET_STRING = "test"
OUTPUT_FILE = f'data/narrativeqa/preprocessed/processed_qaps_{DATASET_STRING}.json'


def generate_question_id(document_id, question):
    """Generate a reproducible hash-based question ID"""
    hash_input = f"{document_id}:{question}".encode('utf-8')
    return hashlib.sha256(hash_input).hexdigest()[:16]  # Truncate for readability

def preprocess_questions(input_file, output_file):
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load CSV
    df = pd.read_csv(input_file)
    
    # Ensure column names are trimmed
    df.columns = df.columns.str.strip()
    
    # Filter only the 'test' set
    df = df[df['set'] == DATASET_STRING]
    
    # Group by document_id
    grouped_data = {}
    for _, row in df.iterrows():
        doc_id = row['document_id']
        question_id = generate_question_id(doc_id, row['question'])
        
        if doc_id not in grouped_data:
            grouped_data[doc_id] = {}
        
        grouped_data[doc_id][question_id] = {
            "question": row['question'],
            "answers": [row['answer1'], row['answer2']],
            "tokenized": {
                "question": row['question_tokenized'],
                "answers": [row['answer1_tokenized'], row['answer2_tokenized']]
            }
        }
    
    # Save to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(grouped_data, f, indent=4, ensure_ascii=False)
    
    print(f'Processed data saved to {output_file}')

if __name__ == "__main__":
    preprocess_questions(INPUT_FILE, OUTPUT_FILE)
