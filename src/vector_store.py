from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Any
from src.config import Config
import uuid

class VectorStore:
    def __init__(self, storage_path: str = Config.QDRANT_STORAGE_PATH, collection_name: str = Config.QDRANT_COLLECTION_NAME):
        self.collection_name = collection_name
        self.client = QdrantClient(path=storage_path)
        
        # Dimensions for BAAI/bge-small-en-v1.5 are 384. 
        # Ideally, we get this dynamically, but for simplicity we hardcode for the default model, 
        # or we create the collection dynamically on first insertion.
        self.vector_size = 384 

    def _ensure_collection(self, vector_size: int):
        """Creates the collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        if not any(c.name == self.collection_name for c in collections):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def index_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """Adds documents and their embeddings to the vector store."""
        if not chunks or not embeddings:
            return
            
        vector_size = len(embeddings[0])
        self._ensure_collection(vector_size)

        points = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid4())
            # Store text in payload alongside metadata
            payload = chunk["metadata"].copy()
            payload["text"] = chunk["text"]
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=emb,
                    payload=payload
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Searches for the most similar documents."""
        self._ensure_collection(len(query_embedding))
        
        search_response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=limit
        )
        
        results = []
        for hit in search_response.points:
            results.append({
                "score": hit.score,
                "text": hit.payload.get("text", ""),
                "metadata": {k: v for k, v in hit.payload.items() if k != "text"}
            })
            
        return results
