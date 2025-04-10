import os
import logging

from openai import OpenAI
from abc import ABC, abstractmethod
from .utils import buildMultipleChoiceQuestionText

from tenacity import retry, stop_after_attempt, wait_exponential, after_log, before_sleep_log

logger = logging.getLogger(__name__)

class BaseQAModel(ABC):
    @abstractmethod
    def answer_question(self, context, question):
        pass

class OpenAI_QAModel_dosRag_MultipleChoice(BaseQAModel):
    def __init__(self, modelString, client):
        """
        Initializes the OpenAI model with the model name set in the modelString

        Args:
            modelName (str): The OpenAI model.
        """
        self.modelString = modelString
        self.client = client

    @retry(wait=wait_exponential(multiplier=1, max=60), 
        stop=stop_after_attempt(10), 
        before_sleep=before_sleep_log(logger, logging.INFO), 
        after=after_log(logger, logging.INFO), 
        reraise=True)
    def answer_question(
        self, context, question, options, tokenizer, max_tokens=150
    ):
        """
        Generates Answers to specified multiple choice questions and options optimized for QuALITY benchmark.
        """
        questionAndOptions = buildMultipleChoiceQuestionText(question, options)

        prompt = f'''
[Start of Context]:

{context}

[End of Context]

[Start of Question]:

{questionAndOptions}

[End of Question]

[Instructions:]
The context contains retrieved text chunks from the original story, marked with their position in the text. Based on the context provided, select the most accurate answer to the question from the given options.
Start with a short explanation and then provide your answer as [[1]] or [[2]] or [[3]] or [[4]]. 
For example, if you think the most accurate answer is the first option, respond with [[1]].
'''
        promptLog = f"\n\n#### Prompting {self.modelString}: ####\n\n{prompt}\n\n#### End of Prompt ####\n\n"
        logging.debug(promptLog)
        # print(promptLog)

        used_input_tokens = len(tokenizer.encode(prompt))

        response = self.client.chat.completions.create(
            model=self.modelString,
            messages=[
                {"role": "system", "content": "You are Question Answering Portal"},
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
            seed = 42,

        )

        answerString = response.choices[0].message.content.strip()
        
        answerLog = f"\n\n#### {self.modelString} Response: ####\n\n{answerString}\n\n#### End of Response ####\n\n"
        logging.debug(answerLog)
        # print(answerLog)

        return answerString, used_input_tokens

class OpenAI_QAModel_MultipleChoice(BaseQAModel):
    def __init__(self, modelString, client):
        """
        Initializes the OpenAI model with the model name set in the modelString

        Args:
            modelName (str): The OpenAI model.
        """
        self.modelString = modelString
        self.client = client

    @retry(wait=wait_exponential(multiplier=1, max=60), 
        stop=stop_after_attempt(10), 
        before_sleep=before_sleep_log(logger, logging.INFO), 
        after=after_log(logger, logging.INFO), 
        reraise=True)
    def answer_question(
        self, context, question, options, tokenizer, max_tokens=150
    ):
        """
        Generates Answers to specified multiple choice questions and options optimized for QuALITY benchmark.
        """
        questionAndOptions = buildMultipleChoiceQuestionText(question, options)

        prompt = f'''
[Start of Context]:

{context}

[End of Context]

[Start of Question]:

{questionAndOptions}

[End of Question]

[Instructions:]
Based on the context provided, select the most accurate answer to the question from the given options.
Start with a short explanation and then provide your answer as [[1]] or [[2]] or [[3]] or [[4]]. 
For example, if you think the most accurate answer is the first option, respond with [[1]].
'''
        promptLog = f"\n\n#### Prompting {self.modelString}: ####\n\n{prompt}\n\n#### End of Prompt ####\n\n"
        logging.debug(promptLog)

        used_input_tokens = len(tokenizer.encode(prompt))

        response = self.client.chat.completions.create(
            model=self.modelString,
            messages=[
                {"role": "system", "content": "You are Question Answering Portal"},
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
            seed = 42,

        )

        answerString = response.choices[0].message.content.strip()
        
        answerLog = f"\n\n#### {self.modelString} Response: ####\n\n{answerString}\n\n#### End of Response ####\n\n"
        logging.debug(answerLog)

        return answerString, used_input_tokens

class OpenAI_QAModel_Generation(BaseQAModel):
    def __init__(self, modelString, client):
        """
        Initializes the OpenAI model with the model name set in the modelString

        Args:
            modelName (str): The OpenAI model.
        """
        self.modelString = modelString
        self.client = client

    @retry(wait=wait_exponential(multiplier=1, max=60), 
        stop=stop_after_attempt(10), 
        before_sleep=before_sleep_log(logger, logging.INFO), 
        after=after_log(logger, logging.INFO), 
        reraise=True)
    def answer_question(
        self, context, question, options, tokenizer, max_tokens=150
    ):
        """
        Generates Answers to specified multiple choice questions and options optimized for QuALITY benchmark.
        """

        prompt = f'''
[Start of Context]:

{context}

[End of Context]

[Start of Question]:

{question}

[End of Question]

[Instructions:]
- Answer the question **only** based on the provided context.
- Keep the answer **short and factual** (preferably between 1-20 words).
- Do **not** provide explanations or additional details beyond what is necessary.
- If the answer is **not explicitly stated** in the context, respond with: "Not found in context."

'''
        promptLog = f"\n\n#### Prompting {self.modelString}: ####\n\n{prompt}\n\n#### End of Prompt ####\n\n"
        logging.debug(promptLog)
        #print(promptLog)

        used_input_tokens = len(tokenizer.encode(prompt))

        response = self.client.chat.completions.create(
            model=self.modelString,
            messages=[
                {"role": "system", "content": "You are Question Answering Portal"},
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
            seed = 42,

        )

        answerString = response.choices[0].message.content.strip()
        
        answerLog = f"\n\n#### {self.modelString} Response: ####\n\n{answerString}\n\n#### End of Response ####\n\n"
        logging.debug(answerLog)
        #print(answerLog)
        
        return answerString, used_input_tokens