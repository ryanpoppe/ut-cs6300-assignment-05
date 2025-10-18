from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

class ShakespeareRAG():
    DB_DIR = Path("./shakespeare_db")
    COLLECTION_NAME = "shakespeare_speeches"
    EMBEDDER = SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/all-MiniLM-L6-v2")

    client = None
    collection = None

    def __init__(self):
        """Initialize the Shakespeare RAG system."""
        self.client = chromadb.PersistentClient(path=str(self.DB_DIR))
        
        try:
            self.collection = self.client.get_collection(
                name=self.COLLECTION_NAME,
                embedding_function=self.EMBEDDER
            )
        except:
            raise RuntimeError(
                "Collection not found. Run 'python -m src.shakespear_rag_extractor' "
                "first to build the database."
            )
        
    def query_shakespeare(self, query_text, n_results=5, filters=None):
        """
        Query the Shakespeare RAG system.
        
        Args:
            query_text: The search query
            n_results: Number of results to return
            filters: Optional metadata filters (e.g., {"play": "Hamlet"})
        
        Returns:
            ChromaDB query results
        """
        query_params = {
            "query_texts": [query_text],
            "n_results": n_results
        }
        
        if filters:
            query_params["where"] = filters
        
        results = self.collection.query(**query_params)
        return results
