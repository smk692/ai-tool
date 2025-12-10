"""Tests for SyncJob model."""

from datetime import UTC, datetime, timedelta

import pytest

from src.models.sync_job import SyncError, SyncJob, SyncJobStatus, SyncJobTrigger


class TestSyncError:
    """Tests for SyncError model."""

    def test_sync_error_creation(self):
        """Test SyncError creation with required fields."""
        error = SyncError(error_type="ValueError", message="Invalid input")
        assert error.error_type == "ValueError"
        assert error.message == "Invalid input"
        assert error.document_id is None
        assert error.source_id is None
        assert error.retryable is False
        assert error.timestamp is not None

    def test_sync_error_with_all_fields(self):
        """Test SyncError creation with all fields."""
        error = SyncError(
            error_type="ConnectionError",
            message="Connection failed",
            document_id="doc-123",
            source_id="src-456",
            retryable=True,
        )
        assert error.document_id == "doc-123"
        assert error.source_id == "src-456"
        assert error.retryable is True


class TestSyncJob:
    """Tests for SyncJob model."""

    def test_sync_job_defaults(self):
        """Test SyncJob creation with defaults."""
        job = SyncJob()
        assert job.id is not None
        assert job.source_id is None
        assert job.trigger == SyncJobTrigger.MANUAL
        assert job.status == SyncJobStatus.PENDING
        assert job.started_at is None
        assert job.completed_at is None
        assert job.documents_processed == 0
        assert job.documents_created == 0
        assert job.documents_updated == 0
        assert job.documents_deleted == 0
        assert job.documents_skipped == 0
        assert job.chunks_created == 0
        assert job.errors == []
        assert job.error_message is None

    def test_sync_job_with_source(self):
        """Test SyncJob creation with source ID."""
        job = SyncJob(source_id="source-123", trigger=SyncJobTrigger.SCHEDULED)
        assert job.source_id == "source-123"
        assert job.trigger == SyncJobTrigger.SCHEDULED

    # ==================== State Methods ====================

    def test_start(self):
        """Test start() method marks job as running."""
        job = SyncJob()
        assert job.status == SyncJobStatus.PENDING
        assert job.started_at is None

        job.start()

        assert job.status == SyncJobStatus.RUNNING
        assert job.started_at is not None
        assert isinstance(job.started_at, datetime)

    def test_complete_success(self):
        """Test complete() method marks job as completed."""
        job = SyncJob()
        job.start()
        job.complete()

        assert job.status == SyncJobStatus.COMPLETED
        assert job.completed_at is not None

    def test_complete_partial(self):
        """Test complete(partial=True) marks job as partial."""
        job = SyncJob()
        job.start()
        job.complete(partial=True)

        assert job.status == SyncJobStatus.PARTIAL
        assert job.completed_at is not None

    def test_fail(self):
        """Test fail() method marks job as failed."""
        job = SyncJob()
        job.start()
        job.fail("Database connection failed")

        assert job.status == SyncJobStatus.FAILED
        assert job.error_message == "Database connection failed"
        assert job.completed_at is not None

    def test_add_error_minimal(self):
        """Test add_error() with minimal fields."""
        job = SyncJob()
        job.add_error(error_type="ValueError", message="Invalid format")

        assert len(job.errors) == 1
        assert job.errors[0].error_type == "ValueError"
        assert job.errors[0].message == "Invalid format"
        assert job.errors[0].document_id is None
        assert job.errors[0].source_id is None
        assert job.errors[0].retryable is False

    def test_add_error_with_all_fields(self):
        """Test add_error() with all optional fields."""
        job = SyncJob()
        job.add_error(
            error_type="ConnectionError",
            message="Connection refused",
            document_id="doc-123",
            source_id="src-456",
            retryable=True,
        )

        assert len(job.errors) == 1
        error = job.errors[0]
        assert error.document_id == "doc-123"
        assert error.source_id == "src-456"
        assert error.retryable is True

    def test_add_multiple_errors(self):
        """Test adding multiple errors."""
        job = SyncJob()
        job.add_error("Error1", "Message1")
        job.add_error("Error2", "Message2")
        job.add_error("Error3", "Message3")

        assert len(job.errors) == 3

    # ==================== Properties ====================

    def test_duration_seconds_completed(self):
        """Test duration_seconds property for completed job."""
        job = SyncJob()
        job.started_at = datetime.now(UTC) - timedelta(seconds=120)
        job.completed_at = datetime.now(UTC)

        duration = job.duration_seconds
        assert duration is not None
        assert duration >= 119  # Allow small timing variance
        assert duration <= 121

    def test_duration_seconds_not_started(self):
        """Test duration_seconds returns None if not started."""
        job = SyncJob()
        assert job.duration_seconds is None

    def test_duration_seconds_not_completed(self):
        """Test duration_seconds returns None if not completed."""
        job = SyncJob()
        job.started_at = datetime.now(UTC)
        assert job.duration_seconds is None

    def test_has_errors_false(self):
        """Test has_errors returns False when no errors."""
        job = SyncJob()
        assert job.has_errors is False

    def test_has_errors_true(self):
        """Test has_errors returns True when errors exist."""
        job = SyncJob()
        job.add_error("Error", "Message")
        assert job.has_errors is True

    # ==================== Serialization ====================

    def test_model_dump_json_safe(self):
        """Test model_dump_json_safe() returns JSON-serializable dict."""
        job = SyncJob()
        job.start()
        job.add_error("TestError", "Test message")
        job.complete()

        data = job.model_dump_json_safe()

        assert isinstance(data["started_at"], str)
        assert isinstance(data["completed_at"], str)
        assert isinstance(data["errors"], list)
        assert isinstance(data["errors"][0]["timestamp"], str)

    def test_model_dump_json_safe_no_timestamps(self):
        """Test model_dump_json_safe() with no timestamps set."""
        job = SyncJob()
        data = job.model_dump_json_safe()

        assert data["started_at"] is None
        assert data["completed_at"] is None

    def test_from_json_safe(self):
        """Test from_json_safe() creates SyncJob from dict."""
        now = datetime.now(UTC)
        data = {
            "id": "job-123",
            "source_id": "src-456",
            "trigger": "scheduled",
            "status": "completed",
            "started_at": now.isoformat(),
            "completed_at": now.isoformat(),
            "documents_processed": 10,
            "errors": [],
        }

        job = SyncJob.from_json_safe(data)

        assert job.id == "job-123"
        assert job.source_id == "src-456"
        assert isinstance(job.started_at, datetime)
        assert isinstance(job.completed_at, datetime)

    def test_from_json_safe_with_errors(self):
        """Test from_json_safe() parses errors correctly."""
        now = datetime.now(UTC)
        data = {
            "id": "job-123",
            "status": "failed",
            "errors": [
                {
                    "error_type": "TestError",
                    "message": "Test message",
                    "document_id": "doc-1",
                    "source_id": "src-1",
                    "timestamp": now.isoformat(),
                    "retryable": False,
                }
            ],
        }

        job = SyncJob.from_json_safe(data)

        assert len(job.errors) == 1
        assert job.errors[0].error_type == "TestError"
        assert isinstance(job.errors[0].timestamp, datetime)

    def test_from_json_safe_no_timestamps(self):
        """Test from_json_safe() handles missing timestamps."""
        data = {
            "id": "job-123",
            "status": "pending",
            "errors": [],
        }

        job = SyncJob.from_json_safe(data)

        assert job.started_at is None
        assert job.completed_at is None

    def test_round_trip_serialization(self):
        """Test that dump and load are inverse operations."""
        original = SyncJob()
        original.start()
        original.add_error("TestError", "Test message", document_id="doc-1")
        original.documents_processed = 50
        original.complete()

        data = original.model_dump_json_safe()
        restored = SyncJob.from_json_safe(data)

        assert restored.id == original.id
        assert restored.status == original.status
        assert restored.documents_processed == original.documents_processed
        assert len(restored.errors) == len(original.errors)


class TestSyncJobEnums:
    """Tests for SyncJob enums."""

    def test_sync_job_status_values(self):
        """Test all SyncJobStatus values."""
        assert SyncJobStatus.PENDING.value == "pending"
        assert SyncJobStatus.RUNNING.value == "running"
        assert SyncJobStatus.COMPLETED.value == "completed"
        assert SyncJobStatus.FAILED.value == "failed"
        assert SyncJobStatus.PARTIAL.value == "partial"

    def test_sync_job_trigger_values(self):
        """Test all SyncJobTrigger values."""
        assert SyncJobTrigger.MANUAL.value == "manual"
        assert SyncJobTrigger.SCHEDULED.value == "scheduled"
