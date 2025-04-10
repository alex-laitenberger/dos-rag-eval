import tiktoken
import nltk
import re

# Ensure the 'punkt' dataset is available
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading 'punkt' dataset...")
    nltk.download('punkt_tab')

def split_text(text: str, tokenizer, max_tokens: int):
    """
    Splits a document into chunks of up to max_tokens.

    Parameters:
        text (str): The input text to split.
        tokenizer: Pre-initialized tiktoken tokenizer.
        max_tokens (int): Maximum tokens per chunk.

    Returns:
        list of str: List of text chunks.
    """

    # Break the text into sentences
    sentences = nltk.tokenize.sent_tokenize(text)

    # Helper function to calculate token count for a sentence
    def get_token_count(sentence):
        return len(tokenizer.encode(" " + sentence))

    # Helper function to handle adding chunks
    def handle_chunk(chunk):
        if chunk:
            chunks.append(" ".join(chunk))

    chunks, current_chunk, current_length = [], [], 0

    for sentence in sentences:
        if not sentence.strip():
            continue

        token_count = get_token_count(sentence)

        # Handle long sentences by further splitting
        if token_count > max_tokens:
            for sub_sentence in re.split(r"(?<=[,;:])\s", sentence):
                sub_token_count = get_token_count(sub_sentence)
                words = sub_sentence.split()

                if sub_token_count > max_tokens:
                    for i in range(0, len(words), max_tokens):
                        handle_chunk(words[i:i + max_tokens])
                else:
                    if current_length + sub_token_count > max_tokens:
                        handle_chunk(current_chunk)
                        current_chunk, current_length = [], 0

                    current_chunk.append(sub_sentence)
                    current_length += sub_token_count
            continue

        # Add sentence to current chunk
        if current_length + token_count > max_tokens:
            handle_chunk(current_chunk)
            current_chunk, current_length = [], 0

        current_chunk.append(sentence)
        current_length += token_count

    handle_chunk(current_chunk)  # Add any remaining chunk
    return chunks


def buildMultipleChoiceQuestionText(questionString, options):
            choicesString = ''
            for index, option in enumerate(options):
                 choicesString += f'[[{index+1}]]: {option} \n'
            return f'{questionString} \nOptions: \n{choicesString}'


def get_text_ordered_with_positions(node_list) -> str:    
    sorted_node_list = sorted(node_list, key=lambda node: node.index)

    previousIndex = -10
    text = ""
    for node in sorted_node_list:
        text += f"[chunk {node.index}] "
        text += f"{' '.join(node.text.splitlines())}"
        #text += f"[end of chunk {node.index}]"
        text += "\n\n"
        previousIndex = node.index

    return text

def get_text_ordered_with_positions_and_gap_indicators(node_list) -> str:    
    sorted_node_list = sorted(node_list, key=lambda node: node.index)

    previousIndex = -10
    text = ""
    for node in sorted_node_list:
        if previousIndex != node.index - 1 and previousIndex != -10:
            chunksInBetween = node.index - previousIndex - 1
            text += f"[... {chunksInBetween} chunks of text in between ...]\n\n"
        text += f"[chunk {node.index}]\n"
        text += f"{' '.join(node.text.splitlines())}\n"
        text += f"[end of chunk {node.index}]"
        text += "\n\n"
        previousIndex = node.index

    return text

def get_text_ordered_with_gap_indicators(node_list) -> str:    
    sorted_node_list = sorted(node_list, key=lambda node: node.index)

    previousIndex = -10
    text = ""
    for node in sorted_node_list:
        if previousIndex != node.index - 1 and previousIndex != -10:
            chunksInBetween = node.index - previousIndex - 1
            #text += f"[... {chunksInBetween} chunks of text in between ...]\n\n"
            text += f"[...]\n\n"
        #text += f"[chunk {node.index}]\n"
        text += f"{' '.join(node.text.splitlines())}"
        #text += f"[end of chunk {node.index}]"
        text += "\n\n"
        previousIndex = node.index

    return text
