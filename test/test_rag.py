import unittest
from source.method.RAG import RAG, Node
from source.method.EmbeddingModels import SnowflakeArcticEmbeddingModel
from source.method.QAModels import OpenAI_QAModel_dosRag_MultipleChoice
from config import OPENAI_API_KEY

from openai import OpenAI

import numpy as np
import torch
import os
import pickle


class TestRAG(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Set up the embedding model and RAG instance once for all tests.
        """
        # Load the API key into the environment
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

        # Initialize the Snowflake Arctic Embedding model (only once)
        openAI_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=0)
        cls.embedding_model = SnowflakeArcticEmbeddingModel()
        cls.qa_model = OpenAI_QAModel_dosRag_MultipleChoice(modelString="gpt-4o-mini-2024-07-18", client=openAI_client)
        cls.rag = RAG(chunk_size=5, embedding_model=cls.embedding_model, qa_model = cls.qa_model)

    def setUp(self):
        """
        Set up a temporary file path for storing nodes.
        """
        self.temp_file = "temp_nodes.pkl"

        # Create sample nodes
        self.rag.nodes = {
            0: Node("This is the first test node.", 0, self.embedding_model.create_embedding("This is the first test node.")),
            1: Node("Another node with some content.", 1, self.embedding_model.create_embedding("Another node with some content.")),
            2: Node("A third test node for the retrieval system.", 2, self.embedding_model.create_embedding("A third test node for the retrieval system.")),
            3: Node("Final node with more data for testing.", 3, self.embedding_model.create_embedding("Final node with more data for testing.")),
        }

    def tearDown(self):
        """
        Clean up the temporary file after each test.
        """
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)

    def test_chunk_and_embed_document_basic(self):
        """
        Test chunk_and_embed_document with a simple document.
        """
        text = "This is a simple test. This should split into chunks."
        
        # Run the method
        nodes = self.rag.chunk_and_embed_document(text)

        # Verify the results
        self.assertEqual(len(nodes), 2)  # Two chunks expected
        self.assertIsInstance(nodes[0], Node)  # Node type check
        self.assertEqual(nodes[0].text, "This is a simple test.")  # Chunk content
        self.assertEqual(nodes[1].text, "This should split into chunks.")  # Chunk content

    def test_chunk_and_embed_document_long_sentence(self):
        """
        Test handling of a document with a long sentence exceeding max_tokens.
        """
        text = "This is a very very very very long sentence that will exceed the max token limit."
        
        # Run the method
        nodes = self.rag.chunk_and_embed_document(text)

        # Verify the results
        self.assertGreater(len(nodes), 1)  # Ensure multiple chunks
        self.assertIsInstance(nodes[0], Node)  # Node type check

    def test_chunk_and_embed_document_empty_text(self):
        """
        Test chunk_and_embed_document with empty text.
        """
        text = ""
        
        # Run the method
        nodes = self.rag.chunk_and_embed_document(text)

        # Verify the results
        self.assertEqual(len(nodes), 0)  # No chunks for empty text

    def test_chunk_and_embed_document_single_word(self):
        """
        Test chunk_and_embed_document with a single word.
        """
        text = "Supercalifragilisticexpialidocious"
        
        # Run the method
        nodes = self.rag.chunk_and_embed_document(text)

        # Verify the results
        self.assertEqual(len(nodes), 1)  # One chunk expected
        self.assertEqual(nodes[0].text, text)  # Check the content of the chunk

    def test_chunk_and_embed_document_mixed_input(self):
        """
        Test chunk_and_embed_document with mixed punctuation and text.
        """
        text = "This is a test, with punctuation; and some random: text."
        
        # Run the method
        nodes = self.rag.chunk_and_embed_document(text)

        # Verify the results
        self.assertGreater(len(nodes), 1)  # Ensure multiple chunks
        self.assertEqual(nodes[0].text, "This is a test,")  # Check the first chunk
        self.assertEqual(nodes[1].text, "with punctuation;")  # Check the second chunk
        self.assertEqual(nodes[2].text, "and some random:")  # Check the third chunk
        self.assertEqual(nodes[3].text, "text.")  # Check the fourth chunk

    def test_chunk_and_embed_document_correct_embedding(self):
        """
        Test that the correct embeddings are generated for each chunk.
        """
        text = "This is a test. This is another sentence."
        
        # Generate nodes
        nodes = self.rag.chunk_and_embed_document(text)
        
        # Verify the embeddings are correct
        for node in nodes.values():
            expected_embedding = self.rag.embedding_model.create_embedding(node.text)
            
            # Check if embeddings match, handling both NumPy arrays and PyTorch tensors
            if isinstance(expected_embedding, np.ndarray):
                np.testing.assert_array_almost_equal(node.embedding, expected_embedding, decimal=6)
            elif isinstance(expected_embedding, torch.Tensor):
                self.assertTrue(torch.allclose(torch.tensor(node.embedding), expected_embedding, atol=1e-6))
            else:
                raise TypeError("Embedding type not supported. Must be NumPy array or PyTorch tensor.")


    def test_embedding_dimensionality(self):
        """
        Test that the dimensionality of the embedding matches the expected value (e.g., 768).
        """
        text = "This is a test sentence."
        expected_dim = 768  # Define the expected dimensionality explicitly

        # Generate nodes
        nodes = self.rag.chunk_and_embed_document(text)

        # Check the dimensionality of each node's embedding
        for node in nodes.values():
            embedding = node.embedding
            if isinstance(embedding, np.ndarray):
                self.assertEqual(embedding.shape[0], expected_dim)
            elif isinstance(embedding, torch.Tensor):
                self.assertEqual(embedding.size(0), expected_dim)
            else:
                raise TypeError("Embedding type not supported. Must be NumPy array or PyTorch tensor.")

    def test_store_and_load_nodes(self):
        """
        Test storing and loading nodes.
        """
        # Create nodes
        text = "This is a test. Another sentence follows."
        self.rag.chunk_and_embed_document(text)

        # Store nodes
        self.rag.store_nodes(self.temp_file)
        self.assertTrue(os.path.exists(self.temp_file))  # Ensure the file is created

        # Clear the current nodes
        self.rag.nodes = None

        # Load nodes
        self.rag.load_nodes(self.temp_file)

        # Verify nodes are correctly loaded
        self.assertEqual(len(self.rag.nodes), 2)  # Two chunks expected
        self.assertIsInstance(self.rag.nodes[0], Node)
        self.assertEqual(self.rag.nodes[0].text, "This is a test.")
        self.assertEqual(self.rag.nodes[1].text, "Another sentence follows.")
        self.assertEqual(len(self.rag.nodes[0].embedding), 768)  # Check embedding dimensionality

    def test_store_nodes_no_nodes(self):
        """
        Test storing nodes when no nodes are present.
        """
        self.rag.nodes = None
        with self.assertRaises(ValueError) as context:
            self.rag.store_nodes(self.temp_file)
        self.assertEqual(str(context.exception), "There are no nodes.")

    def test_load_nodes_invalid_file(self):
        """
        Test loading nodes from an invalid file.
        """
        with open(self.temp_file, "wb") as file:
            pickle.dump("not a node dictionary", file)

        with self.assertRaises(ValueError) as context:
            self.rag.load_nodes(self.temp_file)
        self.assertIn("The loaded object is not a dictionary of nodes", str(context.exception))

    def test_retrieve_basic(self):
        """
        Test basic functionality of retrieve() to ensure it returns relevant nodes.
        """
        query = "test node"
        context, retrieved_node_ids = self.rag.retrieve(query=query, top_k=2, max_tokens=50)

        # Check that the method returns the correct number of nodes
        self.assertEqual(len(retrieved_node_ids), 2)

        # Check that the returned context is a string
        self.assertIsInstance(context, str)

        # Check that the retrieved node IDs are valid
        for node_id in retrieved_node_ids:
            self.assertIn(node_id, self.rag.nodes)

    def test_retrieve_max_tokens(self):
        """
        Test that the retrieve() method respects the max_tokens limit.
        """
        query = "test node"
        max_tokens = 10  # Force a smaller token limit
        context, retrieved_node_ids = self.rag.retrieve(query=query, top_k=10, max_tokens=max_tokens)

        additional_tokens_for_annotations = len(retrieved_node_ids) * 10

        # Verify that the total token count does not exceed the max_tokens limit
        total_tokens = len(self.rag.tokenizer.encode(context))
        self.assertLessEqual(total_tokens, max_tokens+additional_tokens_for_annotations) #respect annotations

    def test_retrieve_no_nodes(self):
        """
        Test that retrieve() raises an error if no nodes are available.
        """
        self.rag.nodes = {}  # Clear the nodes
        query = "test node"

        with self.assertRaises(ValueError) as context:
            self.rag.retrieve(query=query)

        self.assertEqual(str(context.exception), "No nodes available for retrieval.")

    def test_retrieve_top_k(self):
        """
        Test that the retrieve() method returns the correct number of nodes based on top_k.
        """
        query = "test node"
        top_k = 3
        context, retrieved_node_ids = self.rag.retrieve(query=query, top_k=top_k, max_tokens=100)

        # Check that the number of retrieved nodes matches top_k (or fewer if constrained by tokens)
        self.assertLessEqual(len(retrieved_node_ids), top_k)

    def test_retrieve_empty_query(self):
        """
        Test that retrieve() raises an error for an empty query.
        """
        query = ""  # Empty query

        with self.assertRaises(ValueError) as context:
            self.rag.retrieve(query=query)

        self.assertEqual(str(context.exception), "query must not be empty or only whitespace")

    def test_retrieve_invalid_query(self):
        """
        Test that retrieve() raises an error for a non-string query.
        """
        query = 12345  # Invalid query (not a string)

        with self.assertRaises(ValueError) as context:
            self.rag.retrieve(query=query)

        self.assertEqual(str(context.exception), "query must be a string")

    def test_retrieve_invalid_max_tokens(self):
        """
        Test that retrieve() raises an error for invalid max_tokens values.
        """
        query = "test node"

        # Negative max_tokens
        with self.assertRaises(ValueError) as context:
            self.rag.retrieve(query=query, max_tokens=-10)
        self.assertEqual(str(context.exception), "max_tokens must be an integer and at least 1")

        # Non-integer max_tokens
        with self.assertRaises(ValueError) as context:
            self.rag.retrieve(query=query, max_tokens="invalid")
        self.assertEqual(str(context.exception), "max_tokens must be an integer and at least 1")

    def test_retrieve_context_content(self):
        """
        Test that the context generated from the retrieved nodes contains the expected content.
        """
        # Prepare a query and the expected context
        query = "test node"
        expected_context = "This is the first test node. \n\nFinal node with more data for testing."

        # Retrieve context and node IDs
        context, retrieved_node_ids = self.rag.retrieve(query=query, top_k=2, max_tokens=100)

        # Check the context content
        self.assertEqual(context, expected_context)

        # Ensure the node IDs are correctly retrieved
        self.assertEqual(retrieved_node_ids, [0, 3]) 



if __name__ == "__main__":
    unittest.main()
