"""Unit tests for SQLite client."""

import pytest
import sqlite3
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from uuid import uuid4, UUID

from src.db.sqlite import SQLiteClient, SQLiteConversationMemory
from src.utils.errors import DatabaseError


class TestSQLiteClient:
    """Test suite for SQLiteClient class."""

    @pytest.fixture
    def temp_db_path(self, tmp_path):
        """Create temporary database path."""
        return str(tmp_path / "test.db")

    @pytest.fixture
    def sqlite_client(self, temp_db_path):
        """Create SQLiteClient instance with temp database."""
        return SQLiteClient(db_path=temp_db_path)

    @pytest.fixture
    def sample_session_id(self):
        """Create sample session UUID."""
        return uuid4()

    def test_initialization(self, temp_db_path):
        """Test SQLiteClient initialization."""
        client = SQLiteClient(db_path=temp_db_path)

        assert client.db_path == Path(temp_db_path)
        assert client.db_path.parent.exists()
        # Connection is created during initialization for schema setup
        assert client._connection is not None

    def test_initialization_default_path(self):
        """Test SQLiteClient initialization with default path."""
        with patch("src.db.sqlite.settings") as mock_settings:
            mock_settings.sqlite_db_path = "./data/test.db"

            client = SQLiteClient()

            # Path object normalizes to "data/test.db" (removes leading "./")
            assert str(client.db_path) == "data/test.db"

    def test_connection_property_lazy_initialization(self, sqlite_client, temp_db_path):
        """Test connection lazy initialization."""
        # Connection is already created during initialization
        assert sqlite_client._connection is not None

        # Access returns existing connection
        conn = sqlite_client.connection
        assert conn is not None
        assert isinstance(conn, sqlite3.Connection)

        # Second access returns same cached connection
        conn2 = sqlite_client.connection
        assert conn2 is conn

    def test_connection_creates_database_file(self, temp_db_path):
        """Test that connection creates database file."""
        assert not Path(temp_db_path).exists()

        client = SQLiteClient(db_path=temp_db_path)
        _ = client.connection

        assert Path(temp_db_path).exists()

    def test_connection_error_handling(self):
        """Test connection error handling with mock."""
        with patch("src.db.sqlite.sqlite3.connect") as mock_connect:
            # Mock connection to raise a sqlite3.Error
            mock_connect.side_effect = sqlite3.Error("Connection failed")

            # Should raise DatabaseError during initialization
            with pytest.raises(DatabaseError) as exc_info:
                client = SQLiteClient(db_path="/tmp/test.db")

            assert "Failed to establish SQLite connection" in str(exc_info.value)

    def test_schema_initialization(self, sqlite_client):
        """Test database schema is created correctly."""
        conn = sqlite_client.connection
        cursor = conn.cursor()

        # Check sessions table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'"
        )
        assert cursor.fetchone() is not None

        # Check messages table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='messages'"
        )
        assert cursor.fetchone() is not None

        # Check indexes exist
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_messages_session'"
        )
        assert cursor.fetchone() is not None

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_sessions_user'"
        )
        assert cursor.fetchone() is not None

    def test_create_session_success(self, sqlite_client, sample_session_id):
        """Test successful session creation."""
        user_id = "test_user"

        sqlite_client.create_session(sample_session_id, user_id)

        # Verify session was created
        cursor = sqlite_client.connection.cursor()
        cursor.execute(
            "SELECT session_id, user_id FROM sessions WHERE session_id = ?",
            (str(sample_session_id),),
        )
        row = cursor.fetchone()

        assert row is not None
        assert row[0] == str(sample_session_id)
        assert row[1] == user_id

    def test_create_session_duplicate_error(self, sqlite_client, sample_session_id):
        """Test creating duplicate session raises error."""
        user_id = "test_user"

        # Create first session
        sqlite_client.create_session(sample_session_id, user_id)

        # Attempt to create duplicate should raise error
        with pytest.raises(DatabaseError) as exc_info:
            sqlite_client.create_session(sample_session_id, user_id)

        assert "Failed to create session" in str(exc_info.value)

    def test_add_message_user(self, sqlite_client, sample_session_id):
        """Test adding user message."""
        # Create session first
        sqlite_client.create_session(sample_session_id, "test_user")

        # Add user message
        message_id = sqlite_client.add_message(
            session_id=sample_session_id,
            role="user",
            content="Test user message",
        )

        assert isinstance(message_id, int)
        assert message_id > 0

        # Verify message was added
        cursor = sqlite_client.connection.cursor()
        cursor.execute(
            "SELECT role, content FROM messages WHERE message_id = ?",
            (message_id,),
        )
        row = cursor.fetchone()

        assert row is not None
        assert row[0] == "user"
        assert row[1] == "Test user message"

    def test_add_message_assistant(self, sqlite_client, sample_session_id):
        """Test adding assistant message."""
        # Create session first
        sqlite_client.create_session(sample_session_id, "test_user")

        # Add assistant message
        message_id = sqlite_client.add_message(
            session_id=sample_session_id,
            role="assistant",
            content="Test assistant response",
        )

        assert isinstance(message_id, int)

        # Verify message was added
        cursor = sqlite_client.connection.cursor()
        cursor.execute(
            "SELECT role, content FROM messages WHERE message_id = ?",
            (message_id,),
        )
        row = cursor.fetchone()

        assert row[0] == "assistant"
        assert row[1] == "Test assistant response"

    def test_add_message_invalid_role(self, sqlite_client, sample_session_id):
        """Test adding message with invalid role raises error."""
        # Create session first
        sqlite_client.create_session(sample_session_id, "test_user")

        # Attempt to add message with invalid role
        with pytest.raises(DatabaseError) as exc_info:
            sqlite_client.add_message(
                session_id=sample_session_id,
                role="invalid_role",
                content="Test message",
            )

        assert "Invalid role" in str(exc_info.value)

    def test_add_message_updates_session_timestamp(self, sqlite_client, sample_session_id):
        """Test that adding message updates session timestamp."""
        import time

        # Create session
        sqlite_client.create_session(sample_session_id, "test_user")

        # Get initial timestamp
        cursor = sqlite_client.connection.cursor()
        cursor.execute(
            "SELECT updated_at FROM sessions WHERE session_id = ?",
            (str(sample_session_id),),
        )
        initial_timestamp = cursor.fetchone()[0]

        # Wait a bit
        time.sleep(0.1)

        # Add message
        sqlite_client.add_message(
            session_id=sample_session_id,
            role="user",
            content="Test message",
        )

        # Get updated timestamp
        cursor.execute(
            "SELECT updated_at FROM sessions WHERE session_id = ?",
            (str(sample_session_id),),
        )
        updated_timestamp = cursor.fetchone()[0]

        # Timestamp should be updated
        assert updated_timestamp >= initial_timestamp

    def test_get_conversation_history_empty(self, sqlite_client, sample_session_id):
        """Test getting conversation history for non-existent session."""
        history = sqlite_client.get_conversation_history(
            session_id=str(sample_session_id),
            limit=10,
        )

        assert history == []

    def test_get_conversation_history_single_pair(self, sqlite_client, sample_session_id):
        """Test getting conversation history with single user-assistant pair."""
        # Create session and add messages
        sqlite_client.create_session(sample_session_id, "test_user")
        sqlite_client.add_message(sample_session_id, "user", "Hello")
        sqlite_client.add_message(sample_session_id, "assistant", "Hi there")

        # Get history
        history = sqlite_client.get_conversation_history(
            session_id=str(sample_session_id),
            limit=10,
        )

        assert len(history) == 1
        assert history[0]["user_message"] == "Hello"
        assert history[0]["assistant_message"] == "Hi there"

    def test_get_conversation_history_multiple_pairs(self, sqlite_client, sample_session_id):
        """Test getting conversation history with multiple pairs."""
        # Create session and add multiple conversation pairs
        sqlite_client.create_session(sample_session_id, "test_user")

        messages = [
            ("user", "Hello"),
            ("assistant", "Hi there"),
            ("user", "How are you?"),
            ("assistant", "I'm doing well"),
            ("user", "Great!"),
            ("assistant", "Thanks"),
        ]

        for role, content in messages:
            sqlite_client.add_message(sample_session_id, role, content)

        # Get history
        history = sqlite_client.get_conversation_history(
            session_id=str(sample_session_id),
            limit=10,
        )

        assert len(history) == 3
        assert history[0]["user_message"] == "Hello"
        assert history[0]["assistant_message"] == "Hi there"
        assert history[1]["user_message"] == "How are you?"
        assert history[1]["assistant_message"] == "I'm doing well"
        assert history[2]["user_message"] == "Great!"
        assert history[2]["assistant_message"] == "Thanks"

    def test_get_conversation_history_with_limit(self, sqlite_client, sample_session_id):
        """Test conversation history respects limit parameter."""
        # Create session and add many pairs
        sqlite_client.create_session(sample_session_id, "test_user")

        for i in range(10):
            sqlite_client.add_message(sample_session_id, "user", f"Message {i}")
            sqlite_client.add_message(sample_session_id, "assistant", f"Response {i}")

        # Get history with limit
        history = sqlite_client.get_conversation_history(
            session_id=str(sample_session_id),
            limit=3,
        )

        # Should return last 3 pairs (limit * 2 messages retrieved, then last 'limit' pairs)
        assert len(history) == 3
        # The implementation retrieves limit*2 messages (6 messages = 3 pairs from index 0-2)
        # Then returns last 'limit' pairs, which are the first 3 in this case
        assert history[0]["user_message"] == "Message 0"
        assert history[1]["user_message"] == "Message 1"
        assert history[2]["user_message"] == "Message 2"

    def test_get_conversation_history_incomplete_pair(self, sqlite_client, sample_session_id):
        """Test getting conversation history with incomplete pair."""
        # Create session and add user message without assistant response
        sqlite_client.create_session(sample_session_id, "test_user")
        sqlite_client.add_message(sample_session_id, "user", "Hello")

        # Get history
        history = sqlite_client.get_conversation_history(
            session_id=str(sample_session_id),
            limit=10,
        )

        # Should include incomplete pair
        assert len(history) == 1
        assert history[0]["user_message"] == "Hello"
        assert "assistant_message" not in history[0]

    def test_save_conversation_success(self, sqlite_client):
        """Test saving complete conversation pair."""
        session_id = str(uuid4())
        user_message = "What is the weather?"
        assistant_message = "I don't have real-time weather data."

        # Save conversation (creates session automatically)
        sqlite_client.save_conversation(
            session_id=session_id,
            user_message=user_message,
            assistant_message=assistant_message,
        )

        # Verify messages were saved
        history = sqlite_client.get_conversation_history(
            session_id=session_id,
            limit=10,
        )

        assert len(history) == 1
        assert history[0]["user_message"] == user_message
        assert history[0]["assistant_message"] == assistant_message

    def test_save_conversation_with_metadata(self, sqlite_client):
        """Test saving conversation with metadata (metadata is accepted but not used)."""
        session_id = str(uuid4())
        metadata = {"key": "value", "number": 42}

        # Save conversation with metadata (should not raise error)
        sqlite_client.save_conversation(
            session_id=session_id,
            user_message="Test",
            assistant_message="Response",
            metadata=metadata,
        )

        # Verify messages were saved
        history = sqlite_client.get_conversation_history(
            session_id=session_id,
            limit=10,
        )

        assert len(history) == 1

    def test_clear_session_success(self, sqlite_client, sample_session_id):
        """Test clearing session messages."""
        # Create session and add messages
        sqlite_client.create_session(sample_session_id, "test_user")
        sqlite_client.add_message(sample_session_id, "user", "Hello")
        sqlite_client.add_message(sample_session_id, "assistant", "Hi")

        # Verify messages exist
        history_before = sqlite_client.get_conversation_history(
            session_id=str(sample_session_id),
            limit=10,
        )
        assert len(history_before) == 1

        # Clear session
        sqlite_client.clear_session(str(sample_session_id))

        # Verify messages were cleared
        history_after = sqlite_client.get_conversation_history(
            session_id=str(sample_session_id),
            limit=10,
        )
        assert len(history_after) == 0

    def test_clear_session_nonexistent(self, sqlite_client):
        """Test clearing non-existent session (should not raise error)."""
        # Clear non-existent session
        sqlite_client.clear_session(str(uuid4()))

        # Should not raise error

    def test_close_connection(self, sqlite_client):
        """Test closing database connection."""
        # Open connection
        _ = sqlite_client.connection
        assert sqlite_client._connection is not None

        # Close connection
        sqlite_client.close()
        assert sqlite_client._connection is None

    def test_close_already_closed(self, sqlite_client):
        """Test closing already closed connection."""
        # Close without opening
        sqlite_client.close()

        # Should not raise error
        assert sqlite_client._connection is None

    def test_sqlite_conversation_memory_alias(self):
        """Test that SQLiteConversationMemory is an alias for SQLiteClient."""
        assert SQLiteConversationMemory is SQLiteClient


class TestSQLiteClientIntegration:
    """Integration tests for SQLiteClient with real database."""

    @pytest.fixture
    def temp_db_path(self, tmp_path):
        """Create temporary database path."""
        return str(tmp_path / "integration_test.db")

    def test_full_conversation_workflow(self, temp_db_path):
        """Test complete conversation workflow."""
        client = SQLiteClient(db_path=temp_db_path)
        session_id = uuid4()
        user_id = "integration_test_user"

        # Create session
        client.create_session(session_id, user_id)

        # Simulate multi-turn conversation
        conversations = [
            ("Tell me about Python", "Python is a programming language"),
            ("What are its features?", "Python has dynamic typing and automatic memory management"),
            ("Thanks!", "You're welcome!"),
        ]

        for user_msg, assistant_msg in conversations:
            client.save_conversation(
                session_id=str(session_id),
                user_message=user_msg,
                assistant_message=assistant_msg,
            )

        # Retrieve history
        history = client.get_conversation_history(
            session_id=str(session_id),
            limit=10,
        )

        # Verify all conversations were saved
        assert len(history) == 3
        for i, (user_msg, assistant_msg) in enumerate(conversations):
            assert history[i]["user_message"] == user_msg
            assert history[i]["assistant_message"] == assistant_msg

        # Clear and verify
        client.clear_session(str(session_id))
        history_after_clear = client.get_conversation_history(
            session_id=str(session_id),
            limit=10,
        )
        assert len(history_after_clear) == 0

        # Cleanup
        client.close()
