import logging
from source.method.RAG import RAG
from source.method.QAModels import OpenAI_QAModel_Generation
from source.method.EmbeddingModels import SnowflakeArcticEmbeddingModel
from source.experiments.utils import save_jsonl, log_error, create_directories, load_json_file, load_jsonl_file, extract_number
from datetime import datetime
from config import OPENAI_API_KEY
import os

from openai import OpenAI

from concurrent.futures import ThreadPoolExecutor, as_completed

from datetime import datetime

# Load the API key into the environment
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
#OPENAI_MODELSTRING = "gpt-4o-2024-11-20"
OPENAI_MODELSTRING = "gpt-4o-mini-2024-07-18"

#PATHS
STORED_NODES_PATH = "experiments/artifacts/nodes/narrative_qa/test/2025-04-10_13-28-dos-rag-chunk_size_100_Snowflake_Narrative_QA"
STORED_ANSWERS_PATH = "experiments/artifacts/answers/narrative_qa/test"

PREPROCESSED_DATA_PATH = "data/narrativeqa/preprocessed/processed_qaps_test.json"

LOG_DIR = "experiments/logs/"


def get_file_list(folder_path):
    """Get a list of files in a folder, excluding non-files and hidden files."""
    return [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, file)) and not file.startswith(".")
    ]

def run_experiment_on_file(file_path, grouped_dataset, qa_model, embedding_model, hyperparams, stored_answers_file, stored_errors_file):
    """Run the experiment for a single file."""
    document_id = os.path.basename(file_path)  # Extract document ID from filename
    logging.info(f"Processing document: {document_id}")

    # Load the saved nodes
    try:
        # Initialize RetrievalAugmentation with the pre-saved nodes
        rag = RAG(chunk_size=100, embedding_model=embedding_model, qa_model = qa_model)
        rag.load_nodes(file_path)

        questions = grouped_dataset[document_id]

        # Iterate over questions of the document
        for question_id, questionContent in questions.items():
            question = questionContent['question']
            gold_answers = questionContent['answers']

            # Answer the question
            answer, context, node_information, used_input_tokens = rag.answer_question(
                question=question,
                options=None,
                top_k=hyperparams["top_k"],
                max_tokens=hyperparams["max_tokens"]
            )

            if isinstance(answer, str):
                logging.info(
                    f"Document ID: {document_id}, Question ID: {question_id}, Predicted_answer: {answer[:20]}"
                )

                # Store the answer
                result = {
                    "document_id": document_id,
                    "question_id": question_id,
                    "question": question,                    
                    "gold_answers": gold_answers,           
                    "predicted_answer": answer.replace("\n", " "),
                    "node_information": node_information,
                    "used_tokens": used_input_tokens,
                }
                save_jsonl(result, stored_answers_file)
                
            else:
                log_error(document_id, question_id, "No valid string answer", stored_errors_file)

    except Exception as e:
        logging.exception(f"Error processing document {document_id}: {str(e)}")
        save_jsonl({"document_id": document_id, "error": str(e)}, stored_errors_file)

def run_experiment_for_all_files(experiment_identifier, hyperparams):
    """Run a single experiment."""
    current_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    stored_answers_file = f"{STORED_ANSWERS_PATH}/{current_date_time}-{experiment_identifier}.jsonl"
    stored_errors_file = f"{STORED_ANSWERS_PATH}/{current_date_time}-{experiment_identifier}_ERRORS.jsonl"
    log_file = f"{LOG_DIR}/{current_date_time}-{experiment_identifier}.log"

    # Ensure necessary directories exist
    create_directories([STORED_ANSWERS_PATH, LOG_DIR])

     # Logging Configuration
    logger = logging.getLogger()  # Get the root logger
    logger.setLevel(logging.DEBUG)  # Set the general logging level for the root logger, level is set again for the handlers
    
    # Remove existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file, mode="w")  # Write logs to file
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

    logging.info(f"Starting experiment: {experiment_identifier}")

    # Load preprocessed dataset
    grouped_dataset = load_json_file(PREPROCESSED_DATA_PATH)

    # Initialize models
    openAI_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=0)
    qa_model = OpenAI_QAModel_Generation(modelString=OPENAI_MODELSTRING, client=openAI_client)
    embedding_model = SnowflakeArcticEmbeddingModel()

    # Load precreated nodes
    file_list = get_file_list(STORED_NODES_PATH)

    try:
        #with ThreadPoolExecutor(max_workers=1) as executor: #optionally control amount of parallelity
        with ThreadPoolExecutor() as executor:
            logging.info("Using multithreaded run_experiment on all files")
            futures = [
                executor.submit(
                    run_experiment_on_file,
                    file_path, 
                    grouped_dataset, 
                    qa_model, 
                    embedding_model, 
                    hyperparams, 
                    stored_answers_file, 
                    stored_errors_file
                )
                for file_path in file_list
            ]

            for future in as_completed(futures):
                # check if a thread fails with exception
                exception = future.exception()
                if exception:
                    logging.error(f"Error while running experiment on all files: {exception}")

                    # Propagate the exception
                    raise exception

    except Exception as e:
        logging.exception(f"While running experiments the following error ocurred: {e}")

    logging.info(f"Experiment {experiment_identifier} completed.")

def run_experiment_batch():
    """Run a batch of experiments with varying configurations."""
    experiment_tag = "dos-rag-narrative-test"

    experiments = [
        # {
        # "top_k": 30,
        # "max_tokens": 1500,
        # },
        # {
        # "top_k": 100,
        # "max_tokens": 5000,
        # },
        {
        "top_k": 200,
        "max_tokens": 10000,
        },
        # {
        # "top_k": 400,
        # "max_tokens": 20000,
        # },
        # {
        # "top_k": 600,
        # "max_tokens": 30000,
        # },
        # {
        # "top_k": 800,
        # "max_tokens": 40000,
        # }
    ]

    for index, hyperparams in enumerate(experiments):
        experiment_identifier = f"{experiment_tag}_{index}_top-k-{hyperparams['top_k']}_mt-{hyperparams['max_tokens']}_{OPENAI_MODELSTRING}"
        run_experiment_for_all_files(experiment_identifier, hyperparams)
    
    logging.info(f"Experiment-Batch {experiment_tag} with {len(experiments)} experiments completed.")


if __name__ == "__main__":
    run_experiment_batch()
