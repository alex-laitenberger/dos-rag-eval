import logging
from source.method.RAG import RAG
from source.method.QAModels import OpenAI_QAModel_MultipleChoice
from source.method.EmbeddingModels import SnowflakeArcticEmbeddingModel
from source.experiments.utils import save_jsonl, log_error, create_directories, load_json_file, load_jsonl_file, openFileWithUnknownEncoding, count_words, remove_html_tags
from datetime import datetime
from config import OPENAI_API_KEY
import os

import pandas as pd
import re



from openai import OpenAI

# Experiment metadata
EXPERIMENT_IDENTIFIER = "dos-rag-chunk_size_100_Snowflake_Narrative_QA"
CURRENT_DATE_TIME = datetime.now().strftime("%Y-%m-%d_%H-%M")

# Paths
NARRATIVEQA_PATH = '~/narrativeqa'
STORED_NODES_PATH = f"experiments/artifacts/nodes/narrative_qa/test/{CURRENT_DATE_TIME}-{EXPERIMENT_IDENTIFIER}"
LOG_DIR = "experiments/logs/"
LOG_FILE = f"{LOG_DIR}/preprocess_narrative_{CURRENT_DATE_TIME}.log"


# Ensure necessary directories exist
create_directories([STORED_NODES_PATH, LOG_DIR])

# Load the API key into the environment
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

 # Logging Configuration
logger = logging.getLogger()  # Get the root logger
logger.setLevel(logging.DEBUG)  # Set the general logging level for the root logger, level is set again for the handlers

# Remove existing handlers to avoid duplicates
if logger.hasHandlers():
    logger.handlers.clear()

# File handler
file_handler = logging.FileHandler(LOG_FILE, mode="w")  # Write logs to file
file_handler.setLevel(logging.INFO)  # Set log level for the file handler
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Stream handler (for terminal output)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)  # Set log level for the stream handler
stream_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
stream_handler.setFormatter(stream_formatter)
logger.addHandler(stream_handler)

# Hyperparameters
HYPERPARAMS = {
    "chunk_size": 100,  # Chunk size for splitting documents
}

OPENAI_MODELSTRING = "gpt-4o-mini-2024-07-18"



def precreate_nodes():
    logging.info("Starting precreate_nodes process...")

    # Load data
    dataset = 'test' #test, train, valid

    df = pd.read_csv(f'{NARRATIVEQA_PATH}/documents.csv')
    test_df = df[df['set'] == 'test']
    #print(test_df.head(2))

    qaps_df = pd.read_csv(f'{NARRATIVEQA_PATH}/qaps.csv')

    #print(qaps_df.head(1))

    # Initialize models
    logging.info("Initializing models...")
    openAI_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=0)
    qa_model = OpenAI_QAModel_MultipleChoice(modelString="gpt-4o-mini-2024-07-18", client=openAI_client)
    embedding_model = SnowflakeArcticEmbeddingModel()

    # Initialize RAG pipeline
    rag = RAG(chunk_size=HYPERPARAMS["chunk_size"], embedding_model=embedding_model, qa_model=qa_model)

    for index, row_data in test_df.iterrows():
        try:
            document_id = row_data['document_id']
            logging.info(f"Processing document {document_id}...")
            
            # Open and read the file
            file_path = f'{NARRATIVEQA_PATH}/tmp/{document_id}.content'
            expanded_path = os.path.expanduser(file_path)
            content = openFileWithUnknownEncoding(expanded_path)
            if content is not None:
                document_context = remove_html_tags(content).replace('\n', ' ').replace('\t', ' ')
                wordCount = count_words(document_context)
            
                if wordCount > 0:

                    # Chunk and embed the document
                    rag.chunk_and_embed_document(document_context)
                    logging.info(f"Successfully chunked and embedded document {document_id}.")

                    # Save the nodes to the artifacts folder
                    nodes_file_path = f"{STORED_NODES_PATH}/{document_id}"
                    rag.store_nodes(nodes_file_path)
                    logging.info(f"Saved nodes for document {document_id} at {nodes_file_path}.")
                    
                    # Debug node details
                    logging.info(f"Nodes for document_id {document_id}:")
                    nodes_list = list(rag.nodes.values())  # Convert dictionary values to a list

                    # Log the first 3 nodes
                    logging.info("--- First 3 Nodes ---")
                    for node in nodes_list[:3]:  # First 3 nodes
                        logging.info(f"Index: {node.index}, Text: {node.text}, Embedding: {node.embedding[:5]}...")

                    # Log the last 3 nodes
                    logging.info("--- Last 3 Nodes ---")
                    for node in nodes_list[-3:]:  # Last 3 nodes
                        logging.info(f"Index: {node.index}, Text: {node.text}, Embedding: {node.embedding[:5]}...")
        
        except Exception as e:
            # Log errors for this document
            logging.exception(f"Error processing document_id {document_id}: {e}")
            log_error(document_id, "-", str(e), f"{STORED_NODES_PATH}/error_{document_id}")

if __name__ == "__main__":
    precreate_nodes()
