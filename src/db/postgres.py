"""PostgreSQL read-only database client."""

from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from config.settings import settings
from src.utils.errors import DatabaseError
from src.utils.logging import logger


class PostgreSQLClient:
    """
    PostgreSQL read-only client for business database queries.

    Enforces read-only access and provides safe query execution.
    """

    def __init__(self, connection_url: Optional[str] = None):
        """
        Initialize PostgreSQL client.

        Args:
            connection_url: Database connection URL (defaults to settings)
        """
        self.connection_url = connection_url or settings.postgres_url
        self._engine: Optional[Engine] = None

    @property
    def engine(self) -> Engine:
        """Get or create SQLAlchemy engine."""
        if self._engine is None:
            try:
                self._engine = create_engine(
                    self.connection_url,
                    pool_pre_ping=True,  # Verify connections before using
                    pool_size=5,
                    max_overflow=10,
                    echo=settings.log_level == "DEBUG",
                )
                logger.info("PostgreSQL connection established")
            except SQLAlchemyError as e:
                logger.error(f"Failed to connect to PostgreSQL: {e}")
                raise DatabaseError(
                    message="Failed to establish database connection",
                    details={"error": str(e)},
                )
        return self._engine

    def execute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a read-only SQL query.

        Args:
            query: SQL query string (SELECT only)
            params: Optional query parameters for safe substitution

        Returns:
            List of result rows as dictionaries

        Raises:
            DatabaseError: If query execution fails
            ValidationError: If query contains write operations
        """
        # Validate read-only (basic check)
        query_upper = query.strip().upper()
        write_operations = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER"]
        if any(op in query_upper for op in write_operations):
            raise DatabaseError(
                message="Write operations are not allowed (read-only access)",
                details={"query": query},
            )

        try:
            with self.engine.connect() as connection:
                result = connection.execute(text(query), params or {})
                rows = [dict(row._mapping) for row in result]
                logger.info(
                    f"Query executed successfully: {len(rows)} rows returned"
                )
                return rows
        except SQLAlchemyError as e:
            logger.error(f"Query execution failed: {e}")
            raise DatabaseError(
                message="Query execution failed",
                details={"query": query, "error": str(e)},
            )

    def get_schema_info(self) -> Dict[str, List[str]]:
        """
        Retrieve database schema information (tables and columns).

        Returns:
            Dictionary mapping table names to column lists
        """
        schema_query = """
        SELECT
            table_name,
            column_name,
            data_type
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position
        """

        try:
            rows = self.execute_query(schema_query)
            schema = {}
            for row in rows:
                table = row["table_name"]
                if table not in schema:
                    schema[table] = []
                schema[table].append(
                    f"{row['column_name']} ({row['data_type']})"
                )
            logger.info(f"Schema retrieved: {len(schema)} tables")
            return schema
        except DatabaseError:
            logger.error("Failed to retrieve schema information")
            raise

    def close(self) -> None:
        """Close database connection and dispose engine."""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            logger.info("PostgreSQL connection closed")
