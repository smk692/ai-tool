"""SQLite client for conversation memory management."""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

from config.settings import settings
from src.utils.errors import DatabaseError
from src.utils.logging import logger


class SQLiteClient:
    """
    SQLite client for managing conversation history and multi-turn context.

    Stores user sessions, messages, and conversation state.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize SQLite client.

        Args:
            db_path: Path to SQLite database file (defaults to settings)
        """
        self.db_path = Path(db_path or settings.sqlite_db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection: Optional[sqlite3.Connection] = None
        self._initialize_schema()

    @property
    def connection(self) -> sqlite3.Connection:
        """Get or create SQLite connection."""
        if self._connection is None:
            try:
                self._connection = sqlite3.connect(
                    self.db_path,
                    check_same_thread=False,
                )
                self._connection.row_factory = sqlite3.Row  # Dict-like access
                logger.info(f"SQLite connection established: {self.db_path}")
            except sqlite3.Error as e:
                logger.error(f"Failed to connect to SQLite: {e}")
                raise DatabaseError(
                    message="Failed to establish SQLite connection",
                    details={"db_path": str(self.db_path), "error": str(e)},
                )
        return self._connection

    def _initialize_schema(self) -> None:
        """Create tables if they don't exist."""
        schema_sql = """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS messages (
            message_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,  -- 'user' or 'assistant'
            content TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        );

        CREATE INDEX IF NOT EXISTS idx_messages_session
        ON messages(session_id);

        CREATE INDEX IF NOT EXISTS idx_sessions_user
        ON sessions(user_id);
        """

        try:
            cursor = self.connection.cursor()
            cursor.executescript(schema_sql)
            self.connection.commit()
            logger.info("SQLite schema initialized")
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize schema: {e}")
            raise DatabaseError(
                message="Failed to initialize SQLite schema",
                details={"error": str(e)},
            )

    def create_session(self, session_id: UUID, user_id: str) -> None:
        """Create a new conversation session."""
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "INSERT INTO sessions (session_id, user_id) VALUES (?, ?)",
                (str(session_id), user_id),
            )
            self.connection.commit()
            logger.info(f"Session created: {session_id}")
        except sqlite3.Error as e:
            logger.error(f"Failed to create session: {e}")
            raise DatabaseError(
                message="Failed to create session",
                details={"session_id": str(session_id), "error": str(e)},
            )

    def add_message(
        self,
        session_id: UUID,
        role: str,
        content: str,
    ) -> int:
        """
        Add a message to a conversation session.

        Args:
            session_id: Conversation session ID
            role: Message role ('user' or 'assistant')
            content: Message content

        Returns:
            Message ID
        """
        if role not in ("user", "assistant"):
            raise DatabaseError(
                message="Invalid role",
                details={"role": role, "expected": ["user", "assistant"]},
            )

        try:
            cursor = self.connection.cursor()
            cursor.execute(
                """
                INSERT INTO messages (session_id, role, content)
                VALUES (?, ?, ?)
                """,
                (str(session_id), role, content),
            )
            self.connection.commit()

            # Update session timestamp
            cursor.execute(
                """
                UPDATE sessions
                SET updated_at = CURRENT_TIMESTAMP
                WHERE session_id = ?
                """,
                (str(session_id),),
            )
            self.connection.commit()

            message_id = cursor.lastrowid
            logger.info(f"Message added: {message_id} (session: {session_id})")
            return message_id
        except sqlite3.Error as e:
            logger.error(f"Failed to add message: {e}")
            raise DatabaseError(
                message="Failed to add message",
                details={"session_id": str(session_id), "error": str(e)},
            )

    def get_conversation_history(
        self,
        session_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve conversation history for a session as conversation pairs.

        Args:
            session_id: Conversation session ID
            limit: Maximum number of conversation pairs to retrieve

        Returns:
            List of conversation dicts with user_message and assistant_message keys
        """
        try:
            cursor = self.connection.cursor()
            # Get messages ordered by timestamp (oldest first)
            cursor.execute(
                """
                SELECT role, content, timestamp
                FROM messages
                WHERE session_id = ?
                ORDER BY timestamp ASC
                LIMIT ?
                """,
                (session_id, limit * 2),  # Get more messages to ensure pairs
            )
            rows = cursor.fetchall()

            # Group messages into conversation pairs
            conversations = []
            current_conversation = {}

            for row in rows:
                role = row["role"]
                content = row["content"]

                if role == "user":
                    # Start a new conversation pair
                    if current_conversation:
                        conversations.append(current_conversation)
                    current_conversation = {"user_message": content}
                elif role == "assistant" and current_conversation:
                    # Complete the current pair
                    current_conversation["assistant_message"] = content
                    conversations.append(current_conversation)
                    current_conversation = {}

            # Add any incomplete conversation
            if current_conversation:
                conversations.append(current_conversation)

            logger.info(
                f"Retrieved {len(conversations)} conversation pairs for session {session_id}"
            )
            return conversations[-limit:] if len(conversations) > limit else conversations

        except sqlite3.Error as e:
            logger.error(f"Failed to retrieve conversation history: {e}")
            raise DatabaseError(
                message="Failed to retrieve conversation history",
                details={"session_id": session_id, "error": str(e)},
            )

    def save_conversation(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save a user-assistant conversation pair.

        Args:
            session_id: Conversation session ID
            user_message: User's message
            assistant_message: Assistant's response
            metadata: Optional metadata (stored but not used currently)
        """
        session_uuid = UUID(session_id)

        # Add user message
        self.add_message(
            session_id=session_uuid,
            role="user",
            content=user_message,
        )

        # Add assistant message
        self.add_message(
            session_id=session_uuid,
            role="assistant",
            content=assistant_message,
        )

        logger.info(f"Conversation saved for session: {session_id}")

    def clear_session(self, session_id: str) -> None:
        """
        Clear all messages for a session.

        Args:
            session_id: Conversation session ID to clear
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "DELETE FROM messages WHERE session_id = ?",
                (session_id,),
            )
            self.connection.commit()
            logger.info(f"Cleared messages for session: {session_id}")
        except sqlite3.Error as e:
            logger.error(f"Failed to clear session: {e}")
            raise DatabaseError(
                message="Failed to clear session",
                details={"session_id": session_id, "error": str(e)},
            )

    def close(self) -> None:
        """Close database connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None
            logger.info("SQLite connection closed")


# Alias for backward compatibility with multi_turn.py
SQLiteConversationMemory = SQLiteClient
