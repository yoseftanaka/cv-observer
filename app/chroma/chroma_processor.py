import chromadb
import os
from dotenv import load_dotenv
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

load_dotenv()


class ChromaProcessor:
    def __init__(self, collection_name: str):
        self.client = chromadb.PersistentClient(path="data")
        try:
            self.client.delete_collection(collection_name)
        except Exception as e:
            print(
                f"Exception ignored while deleting collection '{collection_name}': {e}"
            )

        self.embedding_func = OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small"
        )

    def get_collection(self, collection_name: str):
        try:
            return self.client.get_collection(
                name=collection_name, embedding_function=self.embedding_func
            )
        except Exception:
            return self.client.create_collection(
                name=collection_name, embedding_function=self.embedding_func
            )

    def query(self, prompt: str, collection_name: str, chunk_size: int, collection):
        collection = self.get_collection(collection_name)
        result = collection.query(query_texts=[prompt], n_results=chunk_size)
        return result
