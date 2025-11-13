"""ChromaDB vector store operations for document retrieval."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import settings
from src.utils.errors import VectorStoreError
from src.utils.logging import logger


class VectorStore:
    """
    ChromaDB vector store client for document embedding and retrieval.

    Handles document indexing, query embedding, and semantic search.
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        """
        Initialize ChromaDB vector store.

        Args:
            persist_directory: Directory for persistent storage
            collection_name: Name of the collection
        """
        self.persist_directory = Path(
            persist_directory or settings.chroma_persist_directory
        )
        self.collection_name = collection_name or settings.chroma_collection_name
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self._client: Optional[chromadb.Client] = None
        self._collection: Optional[chromadb.Collection] = None

    @property
    def client(self) -> chromadb.Client:
        """Get or create ChromaDB client."""
        if self._client is None:
            try:
                self._client = chromadb.Client(
                    ChromaSettings(
                        persist_directory=str(self.persist_directory),
                        anonymized_telemetry=False,
                    )
                )
                logger.info(
                    f"ChromaDB client initialized: {self.persist_directory}"
                )
            except Exception as e:
                logger.error(f"Failed to initialize ChromaDB: {e}")
                raise VectorStoreError(
                    message="Failed to initialize ChromaDB client",
                    details={"error": str(e)},
                )
        return self._client

    @property
    def collection(self) -> chromadb.Collection:
        """Get or create ChromaDB collection."""
        if self._collection is None:
            try:
                self._collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},  # Cosine similarity
                )
                logger.info(f"ChromaDB collection ready: {self.collection_name}")
            except Exception as e:
                logger.error(f"Failed to get/create collection: {e}")
                raise VectorStoreError(
                    message="Failed to access ChromaDB collection",
                    details={"collection": self.collection_name, "error": str(e)},
                )
        return self._collection

    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
        embeddings: Optional[List[List[float]]] = None,
    ) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: List of document texts
            metadatas: List of metadata dictionaries
            ids: List of document IDs
            embeddings: Optional pre-computed embeddings (if None, will be computed)

        Raises:
            VectorStoreError: If document addition fails
        """
        if len(documents) != len(metadatas) != len(ids):
            raise VectorStoreError(
                message="Mismatched lengths",
                details={
                    "documents": len(documents),
                    "metadatas": len(metadatas),
                    "ids": len(ids),
                },
            )

        try:
            if embeddings:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                    embeddings=embeddings,
                )
            else:
                # ChromaDB will compute embeddings using default model
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                )
            logger.info(f"Added {len(documents)} documents to vector store")
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise VectorStoreError(
                message="Failed to add documents to vector store",
                details={"count": len(documents), "error": str(e)},
            )

    def query(
        self,
        query_texts: List[str],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Query the vector store for similar documents.

        Args:
            query_texts: List of query texts
            n_results: Number of results to return per query
            where: Optional metadata filter

        Returns:
            Query results with documents, distances, and metadata

        Raises:
            VectorStoreError: If query fails
        """
        try:
            results = self.collection.query(
                query_texts=query_texts,
                n_results=n_results,
                where=where,
            )
            logger.info(
                f"Vector query executed: {len(query_texts)} queries, "
                f"{n_results} results each"
            )
            return results
        except Exception as e:
            logger.error(f"Vector query failed: {e}")
            raise VectorStoreError(
                message="Vector search query failed",
                details={"error": str(e)},
            )

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.

        Returns:
            Dictionary with collection count and metadata
        """
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": str(self.persist_directory),
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            raise VectorStoreError(
                message="Failed to retrieve collection statistics",
                details={"error": str(e)},
            )

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self._collection = None
            logger.info(f"Collection deleted: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise VectorStoreError(
                message="Failed to delete collection",
                details={"collection": self.collection_name, "error": str(e)},
            )
