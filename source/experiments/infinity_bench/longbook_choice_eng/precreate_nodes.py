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
EXPERIMENT_IDENTIFIER = "dos-rag-chunk_size_100_Snowflake"
CURRENT_DATE_TIME = datetime.now().strftime("%Y-%m-%d_%H-%M")

# Paths
STORED_NODES_PATH = f"experiments/artifacts/nodes/infinity_bench/longbook_choice_eng/{CURRENT_DATE_TIME}-{EXPERIMENT_IDENTIFIER}"
STORED_ANSWERS_PATH = "experiments/artifacts/answers/infinity_bench/longbook_choice_eng/"
LOG_DIR = "experiments/logs/"
LOG_FILE = f"{LOG_DIR}/infinity_bench_longbook_choice_eng_precreate_nodes_{CURRENT_DATE_TIME}.log"


# Ensure necessary directories exist
create_directories([STORED_NODES_PATH, STORED_ANSWERS_PATH, LOG_DIR])

# Load the API key into the environment
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Logging Configuration
logging.basicConfig(
    filename=LOG_FILE,
    filemode="w",  # Overwrite logs each time
    level=logging.INFO,  # Adjust the log level as needed (DEBUG, INFO, WARNING, ERROR)
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Hyperparameters
HYPERPARAMS = {
    "chunk_size": 100,  # Chunk size for splitting documents
}


def precreate_nodes():
    logging.info("Starting precreate_nodes process...")

    # Load preprocessed dataset
    preprocessed_path = "data/infinity_bench/preprocessed/longbook_choice_eng_preprocessed.json"
    grouped_data = load_json_file(preprocessed_path)

    # Initialize models
    logging.info("Initializing models...")
    openAI_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=0)
    qa_model = OpenAI_QAModel_MultipleChoice(modelString="gpt-4o-mini-2024-07-18", client=openAI_client)
    embedding_model = SnowflakeArcticEmbeddingModel()

    # Initialize RAG pipeline
    rag = RAG(chunk_size=HYPERPARAMS["chunk_size"], embedding_model=embedding_model, qa_model=qa_model)

    for doc_id, doc_data in grouped_data.items():
        try:
            logging.info(f"Processing document {doc_id}...")

            # Extract document context
            document_context = doc_data["context"]

            # Chunk and embed the document
            rag.chunk_and_embed_document(document_context)
            logging.info(f"Successfully chunked and embedded document {doc_id}.")

            # Save the nodes to the artifacts folder
            nodes_file_path = f"{STORED_NODES_PATH}/{doc_id}"
            rag.store_nodes(nodes_file_path)
            logging.info(f"Saved nodes for document {doc_id} at {nodes_file_path}.")
            
            # Debug node details
            logging.info(f"Nodes for doc_id {doc_id}:")
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
            log_error(doc_id, "-", str(e), STORED_ANSWERS_PATH)
            print(f"Error processing doc_id {doc_id}: {e}")

if __name__ == "__main__":
    precreate_nodes()
