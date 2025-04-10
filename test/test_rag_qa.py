import unittest
import os
from source.method.RAG import RAG, Node
from source.method.EmbeddingModels import SnowflakeArcticEmbeddingModel
from source.method.QAModels import OpenAI_QAModel_dosRag_MultipleChoice
from config import OPENAI_API_KEY

from openai import OpenAI

class TestRAG(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Set up the embedding model and RAG instance once for all tests.
        """
        # Load the API key into the environment
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

        # Initialize the Snowflake Arctic Embedding model and RAG
        openAI_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=0)
        cls.embedding_model = SnowflakeArcticEmbeddingModel()
        cls.qa_model = OpenAI_QAModel_dosRag_MultipleChoice(modelString="gpt-4o-mini-2024-07-18", client=openAI_client)
        cls.rag = RAG(chunk_size=50, embedding_model=cls.embedding_model, qa_model = cls.qa_model)

    def setUp(self):
        """
        Set up nodes for testing.
        """
        self.rag.nodes = {
            0: Node("This is a document about space exploration.", 0, self.embedding_model.create_embedding("This is a document about space exploration.")),
            1: Node("The moon landing was in 1969.", 1, self.embedding_model.create_embedding("The moon landing was in 1969.")),
            2: Node("Mars is the fourth planet from the Sun.", 2, self.embedding_model.create_embedding("Mars is the fourth planet from the Sun.")),
            3: Node("Satellites orbit the Earth.", 3, self.embedding_model.create_embedding("Satellites orbit the Earth.")),
        }

    def test_answer_question_multiple_choice(self):
        """
        Test the answer_question method for a multiple-choice question.
        """
        question = "What is Mars?"
        options = ["A star", "A planet", "A satellite", "A moon"]

        # Call the answer_question method
        answer, context, node_information, used_input_tokens = self.rag.answer_question(
            question=question,
            options=options,
            top_k=3,
            max_tokens=50
        )

        # Verify the returned answer
        self.assertIn("[[2]]", answer)  # The answer should include the correct option
        self.assertGreater(len(context), 0)  # Context should not be empty
        self.assertIsInstance(node_information, list)  # Node information should be a list
        self.assertTrue(all(isinstance(node_id, int) for node_id in node_information))  # Node IDs should be integers


if __name__ == "__main__":
    unittest.main()
