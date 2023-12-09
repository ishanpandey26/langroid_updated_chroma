import os
from typing import Callable, List

from dotenv import load_dotenv
from openai import OpenAI

from langroid.embedding_models.base import EmbeddingModel, EmbeddingModelsConfig
from langroid.language_models.utils import retry_with_exponential_backoff
from langroid.mytypes import Embeddings


class OpenAIEmbeddingsConfig(EmbeddingModelsConfig):
    model_type: str = "openai"
    model_name: str = "text-embedding-ada-002"
    api_key: str = ""
    organization: str = ""
    dims: int = 1536


class SentenceTransformerEmbeddingsConfig(EmbeddingModelsConfig):
    model_type: str = "sentence-transformer"
    model_name: str = "BAAI/bge-large-en-v1.5"


class OpenAIEmbeddings(EmbeddingModel):
    def __init__(self, config: OpenAIEmbeddingsConfig):
        super().__init__()
        self.config = config
        load_dotenv()
        self.config.api_key = os.getenv("OPENAI_API_KEY", "")
        self.config.organization = os.getenv("OPENAI_ORGANIZATION", "")
        if self.config.api_key == "":
            raise ValueError(
                """OPENAI_API_KEY env variable must be set to use 
                OpenAIEmbeddings. Please set the OPENAI_API_KEY value 
                in your .env file.
                """
            )
        self.client = OpenAI(api_key=self.config.api_key)

    def embedding_fn(self) -> Callable[[List[str]], Embeddings]:
        @retry_with_exponential_backoff
        def fn(texts: List[str]) -> Embeddings:
            result = self.client.embeddings.create(
                input=texts, model=self.config.model_name
            )
            return [d.embedding for d in result.data]

        return fn

    @property
    def embedding_dims(self) -> int:
        return self.config.dims


class EmbeddingFunction:
    def __init__(self, model):
        self.model = model

    def __call__(self, input: List[str]) -> Embeddings:
        return self.model.encode(input, convert_to_numpy=True).tolist()

class SentenceTransformerEmbeddings(EmbeddingModel):
    def __init__(self, config: SentenceTransformerEmbeddingsConfig):
        # Import and error handling
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                """
                To use sentence_transformers embeddings, 
                you must install langroid with the [hf-embeddings] extra, e.g.:
                pip install "langroid[hf-embeddings]"
                """
            )

        super().__init__()
        self.config = config
        self.model = SentenceTransformer(self.config.model_name)

    def embedding_fn(self) -> Callable[[List[str]], Embeddings]:
        return EmbeddingFunction(self.model)

    @property
    def embedding_dims(self) -> int:
        dims = self.model.get_sentence_embedding_dimension()
        if dims is None:
            raise ValueError(
                f"Could not get embedding dimension for model {self.config.model_name}"
            )
        return dims



def embedding_model(embedding_fn_type: str = "openai") -> EmbeddingModel:
    """
    Args:
        embedding_fn_type: "openai" or "sentencetransformer" # others soon
    Returns:
        EmbeddingModel
    """
    if embedding_fn_type == "openai":
        return OpenAIEmbeddings  # type: ignore
    else:  # default sentence transformer
        return SentenceTransformerEmbeddings  # type: ignore
