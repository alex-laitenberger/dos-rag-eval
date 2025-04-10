import hashlib
from collections import defaultdict
from source.experiments.utils import load_jsonl_file, create_directories
import json

# Paths to raw and processed data
RAW_DATA_PATH = "data/infinity_bench/raw/longbook_choice_eng.jsonl"
PROCESSED_DATA_FOLDER_PATH = "data/infinity_bench/preprocessed/"
PROCESSED_DATA_PATH = f"{PROCESSED_DATA_FOLDER_PATH}longbook_choice_eng_preprocessed.json"



# Utility Functions
def generate_document_id(context):
    """Generate a unique hash-based ID for a document."""
    return hashlib.md5(context.encode("utf-8")).hexdigest()

def generate_question_id(doc_id, question):
    """Generate a unique hash-based ID for a question within a document."""
    return hashlib.md5((doc_id + question).encode("utf-8")).hexdigest()

# Preprocessing Function
def preprocess_longbook_choice_eng(dataset):
    """
    Preprocess Infinite Bench dataset:
    - Adds `doc_id` and `question_id`
    - Groups questions by document
    - Counts unique documents
    """
    context_to_id = {}
    processed_data = []
    grouped_data = defaultdict(lambda: {"context": "", "entries": []})

    for example in dataset:
        context = example["context"]
        question = example["input"]
        
        # Generate document ID
        if context not in context_to_id:
            doc_id = generate_document_id(context)
            context_to_id[context] = doc_id
        else:
            doc_id = context_to_id[context]
        
        # Generate question ID
        question_id = generate_question_id(doc_id, question)
        
        # Augment example
        example["doc_id"] = doc_id
        example["question_id"] = question_id
        processed_data.append(example)
        
        # Group questions by document
        grouped_data[doc_id]["context"] = context
        grouped_data[doc_id]["entries"].append({
            "question_id": question_id,
            "input": question,
            "answer": example["answer"],
            "options": example["options"],
        })

    return processed_data, grouped_data, len(context_to_id)

# Gold Label Augmentation
def add_gold_choice(grouped_data):
    """
    Adds `gold_choice` field to entries based on correct answer and options.
    """
    for doc_id, doc_data in grouped_data.items():
        for entry in doc_data["entries"]:
            answer = entry["answer"][0]  # Assume `answer` is a list with one correct answer
            options = entry["options"]

            if answer in options:
                entry["gold_choice"] = options.index(answer) + 1  # 1-based index
            else:
                entry["gold_choice"] = None  # Handle missing answers

    return grouped_data

# Saving Preprocessed Data
def save_preprocessed_dataset(data, output_path):
    """
    Save the preprocessed dataset as a JSON file.
    """
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

def main():
    # Step 1: Load raw dataset
    print("Loading raw dataset...")
    raw_data = load_jsonl_file(RAW_DATA_PATH)

    # Step 2: Preprocess dataset (generate doc_id, question_id, etc.)
    print("Preprocessing dataset...")
    processed_data, grouped_data, num_documents = preprocess_longbook_choice_eng(raw_data)

    # Step 3: Add gold labels
    print("Adding gold labels...")
    grouped_data_with_gold = add_gold_choice(grouped_data)

    # Step 4: Save preprocessed dataset
    print("Saving preprocessed dataset...")

    # Ensure necessary directories exist
    create_directories([PROCESSED_DATA_FOLDER_PATH])

    save_preprocessed_dataset(grouped_data_with_gold, PROCESSED_DATA_PATH)

    # Output statistics
    print(f"Successfully preprocessed {len(processed_data)} questions across {num_documents} unique documents.")
    print(f"Saved preprocessed dataset to {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    main()

