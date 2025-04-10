import unittest
import tiktoken
import nltk
import os

from source.method.utils import split_text


class TestSplitText(unittest.TestCase):
    def setUp(self):
        """Set up tiktoken for testing."""
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def test_basic_splitting(self):
        """Test splitting with a simple text."""
        text = "This is a simple test. This should split into chunks."
        max_tokens = 5
        expected = ["This is a simple test.", "This should split into chunks."]
        result = split_text(text, self.tokenizer, max_tokens)
        self.assertEqual(result, expected)

    def test_long_sentence(self):
        """Test handling of a sentence longer than max_tokens."""
        text = "This is a very very long sentence that exceeds the maximum token limit."
        max_tokens = 5
        expected = [
            "This is a very very", 
            "long sentence that exceeds the", 
            "maximum token limit."
        ]
        result = split_text(text, self.tokenizer, max_tokens)
        self.assertEqual(result, expected)

    def test_empty_text(self):
        """Test with empty text."""
        text = ""
        max_tokens = 5
        expected = []
        result = split_text(text, self.tokenizer, max_tokens)
        self.assertEqual(result, expected)

    def test_single_long_word(self):
        """Test handling of a single word longer than max_tokens."""
        text = "Supercalifragilisticexpialidocious"
        max_tokens = 5
        expected = ["Supercalifragilisticexpialidocious"]
        result = split_text(text, self.tokenizer, max_tokens)
        self.assertEqual(result, expected)

    def test_mixed_input(self):
        """Test handling mixed input with punctuation and whitespace."""
        text = "This is a test, with punctuation; and some random: text."
        max_tokens = 6
        expected = [
            "This is a test,", 
            "with punctuation;", 
            "and some random: text."
        ]
        result = split_text(text, self.tokenizer, max_tokens)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
