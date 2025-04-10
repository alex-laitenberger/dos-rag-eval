import json
import os
import re

from bs4 import BeautifulSoup

def save_jsonl(data, file_path):
    """
    Appends a single dictionary to a JSONL file.
    """
    with open(file_path, "a") as f:
        f.write(json.dumps(data) + "\n")

def log_error(document_id, question_id, error_message, file_path):
    """Log errors during experiment execution."""
    error_data = {
        "documentId": document_id,
        "questionId": question_id,
        "error": error_message,
    }
    with open(file_path, "a") as f:
        f.write(json.dumps(error_data) + "\n")

def create_directories(paths):
    """Create directories if they don't exist."""
    for path in paths:
        os.makedirs(path, exist_ok=True)

def load_json_file(file_path):
    """
    Load a dataset from a standard JSON file.
    """
    with open(file_path, "r") as f:
        return json.load(f)

def load_jsonl_file(file_path):
    """
    Load a dataset from a JSONL file (one JSON object per line).
    """
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

def extract_number(text):
    # Define the regex pattern to match a number between [[ ]]
    pattern = r'\[\[(\d+)\]\]'
    
    # Search for the pattern in the text
    match = re.search(pattern, text)
    
    # If a match is found, return the number as an integer
    if match:
        return int(match.group(1))
    else:
        return None

def openFileWithUnknownEncoding(file_path):
    # Try opening the file with different encodings
    encodings = ['latin1', 'utf-8', 'iso-8859-1', 'windows-1252']
    
    for encoding in encodings:
        try:
            with open(file_path, encoding=encoding) as file:
                content = file.read()
                #print(f"File successfully read with {encoding} encoding.")
                return content
                break
        except UnicodeDecodeError as e:
            print(f"Failed to read file with {encoding} encoding. Error: {e}")
    
    return None

def count_words(text):
    words = text.split()
    return len(words)

def remove_html_tags(html_content):
    # Create a BeautifulSoup object and parse the HTML content
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Get the text content without HTML tags
    text = soup.get_text(separator=' ', strip=True)
    
    return text