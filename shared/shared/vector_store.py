"""Qdrant vector store client wrapper.

Provides high-level interface for vector operations.
"""

from typing import Any, Optional
from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)


class VectorStore:
    """Qdrant vector store wrapper.

    Provides high-level operations for vector storage and retrieval.
    """

    DEFAULT_COLLECTION = "rag_documents"
    DEFAULT_DIMENSION = 1024

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        """Initialize vector store client.

        Args:
            host: Qdrant server host.
            port: Qdrant server port.
            api_key: Optional API key for authentication.
            collection_name: Name of the collection to use.
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name or self.DEFAULT_COLLECTION

        self._client: Optional[QdrantClient] = None
        self._api_key = api_key

    @property
    def client(self) -> QdrantClient:
        """Lazy-load Qdrant client."""
        if self._client is None:
            self._client = QdrantClient(
                host=self.host,
                port=self.port,
                api_key=self._api_key,
            )
        return self._client

    def ensure_collection(
        self,
        dimension: int = DEFAULT_DIMENSION,
        distance: Distance = Distance.COSINE,
    ) -> bool:
        """Ensure collection exists, create if not.

        Args:
            dimension: Vector dimension size.
            distance: Distance metric to use.

        Returns:
            True if collection was created, False if already existed.
        """
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=distance,
                ),
            )
            return True
        return False

    def upsert(
        self,
        points: list[dict[str, Any]],
        batch_size: int = 100,
    ) -> int:
        """Upsert points to the collection.

        Args:
            points: List of point dictionaries with 'id', 'vector', and 'payload'.
            batch_size: Number of points per batch.

        Returns:
            Number of points upserted.
        """
        if not points:
            return 0

        total = 0
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            point_structs = [
                PointStruct(
                    id=self._normalize_id(p["id"]),
                    vector=p["vector"],
                    payload=p.get("payload", {}),
                )
                for p in batch
            ]

            self.client.upsert(
                collection_name=self.collection_name,
                points=point_structs,
            )
            total += len(batch)

        return total

    def delete(
        self,
        point_ids: list[str],
    ) -> int:
        """Delete points by IDs.

        Args:
            point_ids: List of point IDs to delete.

        Returns:
            Number of points deleted.
        """
        if not point_ids:
            return 0

        normalized_ids = [self._normalize_id(pid) for pid in point_ids]

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=normalized_ids,
        )

        return len(point_ids)

    def delete_by_filter(
        self,
        field: str,
        value: Any,
    ) -> bool:
        """Delete points matching a filter condition.

        Args:
            field: Field name to filter on.
            value: Value to match.

        Returns:
            True if deletion was executed.
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key=field, match=MatchValue(value=value))]
            ),
        )
        return True

    def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors.

        Args:
            query_vector: Query embedding vector.
            limit: Maximum number of results.
            score_threshold: Minimum similarity score threshold.
            filter_conditions: Optional filter conditions as {field: value}.

        Returns:
            List of search results with 'id', 'score', and 'payload'.
        """
        search_filter = None
        if filter_conditions:
            must_conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter_conditions.items()
            ]
            search_filter = Filter(must=must_conditions)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=search_filter,
        )

        return [
            {
                "id": str(r.id),
                "score": r.score,
                "payload": r.payload or {},
            }
            for r in results
        ]

    def get_point(self, point_id: str) -> Optional[dict[str, Any]]:
        """Get a single point by ID.

        Args:
            point_id: Point ID to retrieve.

        Returns:
            Point data with 'id', 'vector', and 'payload', or None if not found.
        """
        try:
            results = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[self._normalize_id(point_id)],
                with_vectors=True,
            )
            if results:
                point = results[0]
                return {
                    "id": str(point.id),
                    "vector": point.vector,
                    "payload": point.payload or {},
                }
        except Exception:
            pass
        return None

    def point_exists(self, point_id: str) -> bool:
        """Check if a point exists.

        Args:
            point_id: Point ID to check.

        Returns:
            True if point exists, False otherwise.
        """
        return self.get_point(point_id) is not None

    def count(self, filter_conditions: Optional[dict[str, Any]] = None) -> int:
        """Count points in collection.

        Args:
            filter_conditions: Optional filter conditions as {field: value}.

        Returns:
            Number of points matching the filter (or total if no filter).
        """
        count_filter = None
        if filter_conditions:
            must_conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter_conditions.items()
            ]
            count_filter = Filter(must=must_conditions)

        result = self.client.count(
            collection_name=self.collection_name,
            count_filter=count_filter,
            exact=True,
        )

        return result.count

    def get_collection_info(self) -> dict[str, Any]:
        """Get collection information.

        Returns:
            Dictionary with collection metadata.
        """
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": str(info.status),
        }

    def _normalize_id(self, point_id: str | UUID) -> str:
        """Normalize point ID to string format.

        Args:
            point_id: Point ID (string or UUID).

        Returns:
            Normalized string ID.
        """
        return str(point_id)


# Default store instance (configured lazily)
_default_store: Optional[VectorStore] = None


def get_vector_store(
    host: str = "localhost",
    port: int = 6333,
    api_key: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> VectorStore:
    """Get vector store instance.

    Returns cached default instance if using default parameters.

    Args:
        host: Qdrant server host.
        port: Qdrant server port.
        api_key: Optional API key.
        collection_name: Optional collection name.

    Returns:
        VectorStore instance.
    """
    global _default_store

    is_default = (
        host == "localhost"
        and port == 6333
        and api_key is None
        and collection_name is None
    )

    if is_default:
        if _default_store is None:
            _default_store = VectorStore()
        return _default_store

    return VectorStore(
        host=host,
        port=port,
        api_key=api_key,
        collection_name=collection_name,
    )
