import logging
from source.method.RAG import RAG
from source.method.QAModels import OpenAI_QAModel_MultipleChoice
from source.method.EmbeddingModels import SnowflakeArcticEmbeddingModel
from source.experiments.utils import save_jsonl, log_error, create_directories, load_json_file, load_jsonl_file
from datetime import datetime
from config import OPENAI_API_KEY
import os

from openai import OpenAI

# Experiment metadata
EXPERIMENT_IDENTIFIER = "dos-rag-chunk_size_100_Snowflake_Quality_dev"
CURRENT_DATE_TIME = datetime.now().strftime("%Y-%m-%d_%H-%M")

# Paths
STORED_NODES_PATH = f"experiments/artifacts/nodes/quality/dev/{CURRENT_DATE_TIME}-{EXPERIMENT_IDENTIFIER}"
STORED_ANSWERS_PATH = "experiments/artifacts/answers/quality/"
LOG_DIR = "experiments/logs/"
LOG_FILE = f"{LOG_DIR}/preprocess_quality_{CURRENT_DATE_TIME}.log"


# Ensure necessary directories exist
create_directories([STORED_NODES_PATH, STORED_ANSWERS_PATH, LOG_DIR])

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

    # Load preprocessed dataset
    preprocessed_path = "data/quality/preprocessed/QuALITY.v1.0.1.htmlstripped_dev_preprocessed.json"
    data_dict = load_json_file(preprocessed_path)

    # Initialize models
    logging.info("Initializing models...")
    openAI_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=0)
    qa_model = OpenAI_QAModel_MultipleChoice(modelString="gpt-4o-mini-2024-07-18", client=openAI_client)
    embedding_model = SnowflakeArcticEmbeddingModel()

    # Initialize RAG pipeline
    rag = RAG(chunk_size=HYPERPARAMS["chunk_size"], embedding_model=embedding_model, qa_model=qa_model)

    for article_id, content in data_dict.items():
        try:
            logging.info(f"Processing document {article_id}...")

            # Extract document context
            document_context = content['article']

            # Chunk and embed the document
            rag.chunk_and_embed_document(document_context)
            logging.info(f"Successfully chunked and embedded document {article_id}.")

            # Save the nodes to the artifacts folder
            nodes_file_path = f"{STORED_NODES_PATH}/{article_id}"
            rag.store_nodes(nodes_file_path)
            logging.info(f"Saved nodes for document {article_id} at {nodes_file_path}.")
            
            # Debug node details
            logging.info(f"Sample nodes for article_id {article_id}:")
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
            log_error(article_id, "-", str(e), STORED_ANSWERS_PATH)
            print(f"Error processing article_id {article_id}: {e}")

if __name__ == "__main__":
    precreate_nodes()
