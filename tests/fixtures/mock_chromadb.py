"""Mock ChromaDB client and collection for deterministic testing.

This mock provides in-memory vector storage simulation without requiring
the actual chromadb library or persistent storage.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Union


class MockCollection:
    """Mock ChromaDB collection for testing vector operations.

    Simulates vector storage and similarity search with in-memory state.

    Attributes:
        name: Collection identifier
        metadata: Collection-level metadata
        _documents: Internal storage for document data
    """

    def __init__(self, name: str, metadata: Optional[Dict] = None) -> None:
        """Initialize mock collection.

        Args:
            name: Collection identifier
            metadata: Optional collection-level metadata
        """
        self.name = name
        self.metadata = metadata or {}

        # Internal storage: {id: {embedding, metadata, document}}
        self._documents: Dict[str, Dict[str, Any]] = {}

    def add(
        self,
        ids: Union[str, List[str]],
        embeddings: Optional[Union[List[float], List[List[float]]]] = None,
        metadatas: Optional[Union[Dict, List[Dict]]] = None,
        documents: Optional[Union[str, List[str]]] = None
    ) -> None:
        """Add documents to collection with embeddings.

        Args:
            ids: Single ID or list of document IDs
            embeddings: Optional embedding vectors
            metadatas: Optional metadata dictionaries
            documents: Optional document text content

        Raises:
            ValueError: If list lengths don't match
        """
        # Normalize inputs to lists
        if isinstance(ids, str):
            ids = [ids]
        if embeddings is not None and isinstance(embeddings[0], (int, float)):
            embeddings = [embeddings]
        if metadatas is not None and isinstance(metadatas, dict):
            metadatas = [metadatas]
        if documents is not None and isinstance(documents, str):
            documents = [documents]

        # Validation
        n = len(ids)
        if embeddings is not None and len(embeddings) != n:
            raise ValueError(f"Embedding count ({len(embeddings)}) != ID count ({n})")
        if metadatas is not None and len(metadatas) != n:
            raise ValueError(f"Metadata count ({len(metadatas)}) != ID count ({n})")
        if documents is not None and len(documents) != n:
            raise ValueError(f"Document count ({len(documents)}) != ID count ({n})")

        # Store documents
        for i, doc_id in enumerate(ids):
            self._documents[doc_id] = {
                "embedding": embeddings[i] if embeddings else None,
                "metadata": metadatas[i] if metadatas else None,
                "document": documents[i] if documents else None
            }

    def query(
        self,
        query_embeddings: Union[List[float], List[List[float]]],
        n_results: int = 10,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None,
        include: List[str] = ["embeddings", "metadatas", "documents", "distances"]
    ) -> Dict[str, List]:
        """Query collection for similar documents.

        Performs cosine similarity search against stored embeddings.

        Args:
            query_embeddings: Query vectors (single or batch)
            n_results: Maximum results per query (default: 10)
            where: Optional metadata filter (not implemented in mock)
            where_document: Optional document filter (not implemented in mock)
            include: Fields to include in results

        Returns:
            Dict with keys: ids, distances, documents, metadatas, embeddings
            Each value is a list of lists (one list per query)

        Raises:
            ValueError: If collection is empty
        """
        # Normalize query to batch format
        if not isinstance(query_embeddings[0], list):
            query_embeddings = [query_embeddings]

        if not self._documents:
            raise ValueError("Cannot query empty collection")

        # Initialize result structure
        results = {
            "ids": [],
            "distances": [],
            "documents": [],
            "metadatas": [],
            "embeddings": []
        }

        # Process each query
        for query_emb in query_embeddings:
            query_vec = np.array(query_emb, dtype=np.float32)

            # Calculate distances to all documents
            similarities = []
            for doc_id, doc_data in self._documents.items():
                if doc_data["embedding"] is None:
                    continue

                doc_vec = np.array(doc_data["embedding"], dtype=np.float32)

                # Euclidean distance (L2) - simpler and more reliable for constant vectors
                distance = float(np.linalg.norm(query_vec - doc_vec))

                similarities.append((doc_id, distance, doc_data))

            # Sort by distance (ascending) and take top n_results
            similarities.sort(key=lambda x: x[1])
            top_results = similarities[:n_results]

            # Build result lists for this query
            query_ids = [doc_id for doc_id, _, _ in top_results]
            query_distances = [dist for _, dist, _ in top_results]
            query_documents = [data["document"] for _, _, data in top_results]
            query_metadatas = [data["metadata"] for _, _, data in top_results]
            query_embeddings = [data["embedding"] for _, _, data in top_results]

            results["ids"].append(query_ids)
            results["distances"].append(query_distances)
            results["documents"].append(query_documents)
            results["metadatas"].append(query_metadatas)
            results["embeddings"].append(query_embeddings)

        # Filter based on include parameter
        return {k: v for k, v in results.items() if k in include or k == "ids"}

    def get(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        include: List[str] = ["embeddings", "metadatas", "documents"]
    ) -> Dict[str, List]:
        """Get documents by ID or filter.

        Args:
            ids: Optional list of IDs to retrieve
            where: Optional metadata filter (not implemented in mock)
            limit: Maximum results to return
            offset: Result offset for pagination
            include: Fields to include in results

        Returns:
            Dict with requested fields
        """
        if ids is None:
            ids = list(self._documents.keys())

        # Apply pagination
        if offset:
            ids = ids[offset:]
        if limit:
            ids = ids[:limit]

        # Build results
        results = {
            "ids": ids,
            "embeddings": [],
            "metadatas": [],
            "documents": []
        }

        for doc_id in ids:
            if doc_id in self._documents:
                doc_data = self._documents[doc_id]
                results["embeddings"].append(doc_data["embedding"])
                results["metadatas"].append(doc_data["metadata"])
                results["documents"].append(doc_data["document"])

        # Filter based on include parameter
        return {k: v for k, v in results.items() if k in include or k == "ids"}

    def count(self) -> int:
        """Get number of documents in collection.

        Returns:
            int: Document count
        """
        return len(self._documents)

    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict] = None
    ) -> None:
        """Delete documents from collection.

        Args:
            ids: Optional list of IDs to delete
            where: Optional metadata filter (not implemented in mock)
        """
        if ids is None:
            # Delete all
            self._documents.clear()
        else:
            for doc_id in ids:
                self._documents.pop(doc_id, None)


class MockChromaClient:
    """Mock ChromaDB client for testing.

    Simulates ChromaDB client with in-memory collection management.

    Attributes:
        _collections: Internal storage for collections
    """

    def __init__(self) -> None:
        """Initialize mock ChromaDB client."""
        self._collections: Dict[str, MockCollection] = {}

    def get_or_create_collection(
        self,
        name: str,
        metadata: Optional[Dict] = None,
        embedding_function: Optional[Any] = None
    ) -> MockCollection:
        """Get existing collection or create new one.

        Args:
            name: Collection identifier
            metadata: Optional collection-level metadata
            embedding_function: Optional embedding function (ignored in mock)

        Returns:
            MockCollection: Collection instance
        """
        if name not in self._collections:
            self._collections[name] = MockCollection(name, metadata)

        return self._collections[name]

    def create_collection(
        self,
        name: str,
        metadata: Optional[Dict] = None,
        embedding_function: Optional[Any] = None
    ) -> MockCollection:
        """Create new collection.

        Args:
            name: Collection identifier
            metadata: Optional collection-level metadata
            embedding_function: Optional embedding function (ignored in mock)

        Returns:
            MockCollection: New collection instance

        Raises:
            ValueError: If collection already exists
        """
        if name in self._collections:
            raise ValueError(f"Collection '{name}' already exists")

        collection = MockCollection(name, metadata)
        self._collections[name] = collection
        return collection

    def get_collection(self, name: str) -> MockCollection:
        """Get existing collection.

        Args:
            name: Collection identifier

        Returns:
            MockCollection: Collection instance

        Raises:
            ValueError: If collection doesn't exist
        """
        if name not in self._collections:
            raise ValueError(f"Collection '{name}' not found")

        return self._collections[name]

    def delete_collection(self, name: str) -> None:
        """Delete collection.

        Args:
            name: Collection identifier to delete
        """
        self._collections.pop(name, None)

    def list_collections(self) -> List[MockCollection]:
        """List all collections.

        Returns:
            List[MockCollection]: All collection instances
        """
        return list(self._collections.values())

    def reset(self) -> None:
        """Reset client by deleting all collections."""
        self._collections.clear()
