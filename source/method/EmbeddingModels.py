from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer


class BaseEmbeddingModel(ABC):
    @abstractmethod
    def create_embedding(self, text):
        pass
    def create_query_embedding(self, text):
        pass

class SnowflakeArcticEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="Snowflake/snowflake-arctic-embed-m-v1.5"):
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

    def create_embedding(self, text):
        return self.model.encode(text)

    def create_query_embedding(self, text):
        return self.model.encode(text, prompt_name="query")