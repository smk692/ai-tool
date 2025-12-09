"""ChromaDB vector store operations for document retrieval."""

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import settings
from src.services.embeddings import HuggingFaceEmbedding
from src.utils.errors import VectorStoreError
from src.utils.logging import logger


class VectorStore:
    """
    ChromaDB vector store client for document embedding and retrieval.

    Handles document indexing, query embedding, and semantic search.
    Integrates with HuggingFaceEmbedding for custom embedding generation.
    """

    def __init__(
        self,
        embedding_service: HuggingFaceEmbedding,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        """
        Initialize ChromaDB vector store with embedding service.

        Args:
            embedding_service: HuggingFaceEmbedding service for generating embeddings
            persist_directory: Directory for persistent storage
            collection_name: Name of the collection
        """
        self.embedding_service = embedding_service
        self.persist_directory = Path(
            persist_directory or settings.chroma_persist_directory
        )
        self.collection_name = collection_name or settings.chroma_collection_name
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self._client: Optional[chromadb.Client] = None
        self._collection: Optional[chromadb.Collection] = None

        logger.info(
            f"Initialized VectorStore with collection={self.collection_name}, "
            f"embedding_dim={self.embedding_service.get_embedding_dimension()}"
        )

    @property
    def client(self) -> chromadb.Client:
        """Get or create ChromaDB client."""
        if self._client is None:
            try:
                self._client = chromadb.PersistentClient(
                    path=str(self.persist_directory)
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
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
    ) -> Dict[str, Any]:
        """
        Add documents to the vector store.

        Args:
            documents: List of document texts
            metadatas: Optional list of metadata dictionaries
            ids: Optional list of document IDs (auto-generated if None)
            embeddings: Optional pre-computed embeddings (auto-generated if None)

        Returns:
            Dict with success status, count, and IDs

        Raises:
            VectorStoreError: If document addition fails
        """
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]

        # Generate default metadatas if not provided
        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in documents]

        # Validate lengths
        if not (len(documents) == len(metadatas) == len(ids)):
            raise VectorStoreError(
                message="Mismatched lengths",
                details={
                    "documents": len(documents),
                    "metadatas": len(metadatas),
                    "ids": len(ids),
                },
            )

        try:
            # Generate embeddings using HuggingFaceEmbedding service if not provided
            if embeddings is None:
                logger.info(f"Generating embeddings for {len(documents)} documents")
                embeddings = self.embedding_service.embed_texts(documents)

            # Add to ChromaDB
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings,
            )
            logger.info(f"Added {len(documents)} documents to collection")

            return {
                "success": True,
                "count": len(documents),
                "ids": ids
            }

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise VectorStoreError(
                message="Failed to add documents to vector store",
                details={"count": len(documents), "error": str(e)},
            )

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Query the vector store for similar documents.

        Args:
            query_text: Single query text
            top_k: Number of results to return
            filter: Optional metadata filter

        Returns:
            Dict with documents, metadatas, distances, and ids

        Raises:
            VectorStoreError: If query fails
        """
        try:
            # Generate query embedding using HuggingFaceEmbedding service
            logger.debug(f"Generating embedding for query: {query_text[:50]}...")
            query_embedding = self.embedding_service.embed_text(query_text)

            # ChromaDB search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter,
            )

            logger.info(f"Query returned {len(results['documents'][0])} results")

            # Flatten results since we only have one query
            return {
                "documents": results["documents"][0],
                "metadatas": results["metadatas"][0],
                "distances": results["distances"][0],
                "ids": results["ids"][0]
            }

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
