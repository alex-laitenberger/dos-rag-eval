import json
import itertools

from source.experiments.utils import create_directories

# Paths to raw and processed data
RAW_DATA_PATH = "data/quality/raw/QuALITY.v1.0.1.htmlstripped.dev"
PROCESSED_DATA_FOLDER_PATH = "data/quality/preprocessed/"
PROCESSED_DATA_PATH = f"{PROCESSED_DATA_FOLDER_PATH}QuALITY.v1.0.1.htmlstripped_dev_preprocessed.json"


def process_jsonl_file(quality_file_path):
    """
    Processes a JSONL file and extracts relevant information into a dictionary.
    
    :param quality_file_path: Path to the JSONL file.
    :return: Dictionary with processed data.
    """
    result = {}
    
    with open(quality_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            article_id = data["article_id"]
            set_unique_id = data["set_unique_id"]
            article = data["article"]  # No need to remove HTML tags anymore
            questions = data["questions"]
            
            if article_id not in result:
                result[article_id] = {
                    "article": article,
                    "questions": {}
                }
                
            result[article_id]["questions"][set_unique_id] = questions
    
    return result

if __name__ == "__main__":
    
    # Process the file and extract the information
    data_dict = process_jsonl_file(RAW_DATA_PATH)

    # Ensure necessary directories exist
    create_directories([PROCESSED_DATA_FOLDER_PATH])

    # Save the dictionary as a JSON file
    with open(PROCESSED_DATA_PATH, 'w', encoding='utf-8') as json_file:
        json.dump(data_dict, json_file, ensure_ascii=False, indent=4)

    print(f"Processed {len(data_dict)} articles and saved to {PROCESSED_DATA_PATH}")
