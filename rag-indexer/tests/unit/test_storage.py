"""Tests for storage module."""

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from src.models import Document, NotionSourceConfig, Source, SourceType, SyncJob, SyncJobStatus
from src.storage import Storage, get_storage


class TestStorage:
    """Tests for Storage class."""

    @pytest.fixture
    def temp_storage(self, tmp_path):
        """Create a temporary storage instance."""
        return Storage(tmp_path)

    @pytest.fixture
    def sample_source(self):
        """Create a sample source."""
        return Source(
            name="Test Source",
            source_type=SourceType.NOTION,
            config=NotionSourceConfig(
                page_ids=["page-1"],
                database_ids=["db-1"],
            ),
        )

    @pytest.fixture
    def sample_document(self, sample_source):
        """Create a sample document."""
        return Document(
            source_id=sample_source.id,
            external_id="ext-123",
            title="Test Document",
            content="Test content",
            url="https://example.com/doc",
            content_hash="abc123",
        )

    @pytest.fixture
    def sample_sync_job(self, sample_source):
        """Create a sample sync job."""
        return SyncJob(
            source_id=sample_source.id,
            status=SyncJobStatus.RUNNING,
            started_at=datetime.now(UTC),
        )

    # ==================== Initialization ====================

    def test_initialization_creates_directory(self, tmp_path):
        """Test storage creates data directory if not exists."""
        data_dir = tmp_path / "nonexistent"
        storage = Storage(data_dir)
        assert data_dir.exists()
        assert storage.data_dir == data_dir

    def test_initialization_with_string_path(self, tmp_path):
        """Test storage accepts string path."""
        data_dir = str(tmp_path / "string_path")
        storage = Storage(data_dir)
        assert Path(data_dir).exists()

    # ==================== Sources ====================

    def test_get_sources_empty(self, temp_storage):
        """Test get_sources returns empty list when no sources."""
        sources = temp_storage.get_sources()
        assert sources == []

    def test_add_source(self, temp_storage, sample_source):
        """Test adding a source."""
        result = temp_storage.add_source(sample_source)
        assert result.id == sample_source.id
        assert result.name == sample_source.name

        # Verify persisted
        sources = temp_storage.get_sources()
        assert len(sources) == 1
        assert sources[0].id == sample_source.id

    def test_add_source_duplicate_name_raises(self, temp_storage, sample_source):
        """Test adding source with duplicate name raises ValueError."""
        temp_storage.add_source(sample_source)

        duplicate = Source(
            name=sample_source.name,  # Same name
            source_type=SourceType.SWAGGER,
            config={"url": "https://example.com/swagger.json"},
        )

        with pytest.raises(ValueError, match="already exists"):
            temp_storage.add_source(duplicate)

    def test_get_source_by_id(self, temp_storage, sample_source):
        """Test getting source by ID."""
        temp_storage.add_source(sample_source)

        result = temp_storage.get_source(sample_source.id)
        assert result is not None
        assert result.id == sample_source.id

    def test_get_source_by_id_not_found(self, temp_storage):
        """Test getting non-existent source returns None."""
        result = temp_storage.get_source("nonexistent-id")
        assert result is None

    def test_get_source_by_name(self, temp_storage, sample_source):
        """Test getting source by name."""
        temp_storage.add_source(sample_source)

        result = temp_storage.get_source_by_name(sample_source.name)
        assert result is not None
        assert result.name == sample_source.name

    def test_get_source_by_name_not_found(self, temp_storage):
        """Test getting source by non-existent name returns None."""
        result = temp_storage.get_source_by_name("nonexistent-name")
        assert result is None

    def test_get_sources_by_type(self, temp_storage):
        """Test filtering sources by type."""
        notion_source = Source(
            name="Notion Source",
            source_type=SourceType.NOTION,
            config=NotionSourceConfig(page_ids=["p1"]),
        )
        swagger_source = Source(
            name="Swagger Source",
            source_type=SourceType.SWAGGER,
            config={"url": "https://example.com/swagger.json"},
        )

        temp_storage.add_source(notion_source)
        temp_storage.add_source(swagger_source)

        notion_sources = temp_storage.get_sources_by_type(SourceType.NOTION)
        assert len(notion_sources) == 1
        assert notion_sources[0].source_type == SourceType.NOTION

        swagger_sources = temp_storage.get_sources_by_type(SourceType.SWAGGER)
        assert len(swagger_sources) == 1
        assert swagger_sources[0].source_type == SourceType.SWAGGER

    def test_update_source(self, temp_storage, sample_source):
        """Test updating a source."""
        temp_storage.add_source(sample_source)

        sample_source.name = "Updated Name"
        result = temp_storage.update_source(sample_source)

        assert result.name == "Updated Name"
        assert result.updated_at is not None

        # Verify persisted
        stored = temp_storage.get_source(sample_source.id)
        assert stored.name == "Updated Name"

    def test_update_source_not_found_raises(self, temp_storage, sample_source):
        """Test updating non-existent source raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            temp_storage.update_source(sample_source)

    def test_delete_source(self, temp_storage, sample_source):
        """Test deleting a source."""
        temp_storage.add_source(sample_source)

        result = temp_storage.delete_source(sample_source.id)
        assert result is True

        # Verify deleted
        assert temp_storage.get_source(sample_source.id) is None

    def test_delete_source_not_found(self, temp_storage):
        """Test deleting non-existent source returns False."""
        result = temp_storage.delete_source("nonexistent-id")
        assert result is False

    def test_delete_source_removes_related_documents(
        self, temp_storage, sample_source, sample_document
    ):
        """Test deleting source also deletes related documents."""
        temp_storage.add_source(sample_source)
        temp_storage.upsert_document(sample_document)

        temp_storage.delete_source(sample_source.id)

        # Documents should be deleted
        docs = temp_storage.get_documents(sample_source.id)
        assert len(docs) == 0

    # ==================== Documents ====================

    def test_get_documents_empty(self, temp_storage):
        """Test get_documents returns empty list when no documents."""
        docs = temp_storage.get_documents()
        assert docs == []

    def test_upsert_document_new(self, temp_storage, sample_document):
        """Test inserting a new document."""
        result = temp_storage.upsert_document(sample_document)
        assert result.id == sample_document.id

        # Verify persisted
        docs = temp_storage.get_documents()
        assert len(docs) == 1

    def test_upsert_document_update(self, temp_storage, sample_document):
        """Test updating an existing document."""
        temp_storage.upsert_document(sample_document)

        sample_document.title = "Updated Title"
        result = temp_storage.upsert_document(sample_document)

        assert result.title == "Updated Title"
        assert result.updated_at is not None

        # Verify only one document
        docs = temp_storage.get_documents()
        assert len(docs) == 1
        assert docs[0].title == "Updated Title"

    def test_get_documents_filter_by_source(self, temp_storage):
        """Test filtering documents by source_id."""
        source1_id = str(uuid4())
        source2_id = str(uuid4())

        doc1 = Document(
            source_id=source1_id,
            external_id="ext-1",
            title="Doc 1",
            content="Content 1",
        )
        doc2 = Document(
            source_id=source2_id,
            external_id="ext-2",
            title="Doc 2",
            content="Content 2",
        )

        temp_storage.upsert_document(doc1)
        temp_storage.upsert_document(doc2)

        source1_docs = temp_storage.get_documents(source1_id)
        assert len(source1_docs) == 1
        assert source1_docs[0].source_id == source1_id

    def test_get_document_by_id(self, temp_storage, sample_document):
        """Test getting document by ID."""
        temp_storage.upsert_document(sample_document)

        result = temp_storage.get_document(sample_document.id)
        assert result is not None
        assert result.id == sample_document.id

    def test_get_document_by_id_not_found(self, temp_storage):
        """Test getting non-existent document returns None."""
        result = temp_storage.get_document("nonexistent-id")
        assert result is None

    def test_get_document_by_external_id(self, temp_storage, sample_document):
        """Test getting document by external ID."""
        temp_storage.upsert_document(sample_document)

        result = temp_storage.get_document_by_external_id(
            sample_document.source_id,
            sample_document.external_id,
        )
        assert result is not None
        assert result.external_id == sample_document.external_id

    def test_get_document_by_external_id_not_found(self, temp_storage, sample_document):
        """Test getting document by non-existent external ID."""
        temp_storage.upsert_document(sample_document)

        result = temp_storage.get_document_by_external_id(
            sample_document.source_id,
            "nonexistent-ext-id",
        )
        assert result is None

    def test_delete_document(self, temp_storage, sample_document):
        """Test deleting a document."""
        temp_storage.upsert_document(sample_document)

        result = temp_storage.delete_document(sample_document.id)
        assert result is True

        # Verify deleted
        assert temp_storage.get_document(sample_document.id) is None

    def test_delete_document_not_found(self, temp_storage):
        """Test deleting non-existent document returns False."""
        result = temp_storage.delete_document("nonexistent-id")
        assert result is False

    def test_delete_documents_by_source(self, temp_storage):
        """Test deleting all documents for a source."""
        source_id = str(uuid4())

        for i in range(3):
            doc = Document(
                source_id=source_id,
                external_id=f"ext-{i}",
                title=f"Doc {i}",
                content=f"Content {i}",
            )
            temp_storage.upsert_document(doc)

        deleted = temp_storage.delete_documents_by_source(source_id)
        assert deleted == 3

        # Verify all deleted
        docs = temp_storage.get_documents(source_id)
        assert len(docs) == 0

    def test_delete_documents_by_source_none_found(self, temp_storage):
        """Test deleting documents for non-existent source returns 0."""
        deleted = temp_storage.delete_documents_by_source("nonexistent-source")
        assert deleted == 0

    # ==================== Sync History ====================

    def test_get_sync_history_empty(self, temp_storage):
        """Test get_sync_history returns empty list when no jobs."""
        jobs = temp_storage.get_sync_history()
        assert jobs == []

    def test_add_sync_job(self, temp_storage, sample_sync_job):
        """Test adding a sync job."""
        result = temp_storage.add_sync_job(sample_sync_job)
        assert result.id == sample_sync_job.id

        # Verify persisted
        jobs = temp_storage.get_sync_history()
        assert len(jobs) == 1

    def test_sync_history_sorted_by_started_at(self, temp_storage):
        """Test sync history is sorted newest first."""
        source_id = str(uuid4())

        job1 = SyncJob(
            source_id=source_id,
            status=SyncJobStatus.COMPLETED,
            started_at=datetime(2024, 1, 1, tzinfo=UTC),
        )
        job2 = SyncJob(
            source_id=source_id,
            status=SyncJobStatus.COMPLETED,
            started_at=datetime(2024, 1, 3, tzinfo=UTC),
        )
        job3 = SyncJob(
            source_id=source_id,
            status=SyncJobStatus.COMPLETED,
            started_at=datetime(2024, 1, 2, tzinfo=UTC),
        )

        temp_storage.add_sync_job(job1)
        temp_storage.add_sync_job(job2)
        temp_storage.add_sync_job(job3)

        jobs = temp_storage.get_sync_history()
        assert jobs[0].id == job2.id  # Newest first
        assert jobs[1].id == job3.id
        assert jobs[2].id == job1.id

    def test_sync_history_limit(self, temp_storage):
        """Test sync history respects limit parameter."""
        source_id = str(uuid4())

        for i in range(5):
            job = SyncJob(
                source_id=source_id,
                status=SyncJobStatus.COMPLETED,
                started_at=datetime(2024, 1, i + 1, tzinfo=UTC),
            )
            temp_storage.add_sync_job(job)

        jobs = temp_storage.get_sync_history(limit=3)
        assert len(jobs) == 3

    def test_sync_history_max_limit(self, temp_storage):
        """Test sync history respects MAX_SYNC_HISTORY."""
        source_id = str(uuid4())

        # Add more than MAX_SYNC_HISTORY jobs
        for i in range(Storage.MAX_SYNC_HISTORY + 10):
            job = SyncJob(
                source_id=source_id,
                status=SyncJobStatus.COMPLETED,
                started_at=datetime.now(UTC),
            )
            temp_storage.add_sync_job(job)

        jobs = temp_storage.get_sync_history()
        assert len(jobs) <= Storage.MAX_SYNC_HISTORY

    def test_update_sync_job(self, temp_storage, sample_sync_job):
        """Test updating a sync job."""
        temp_storage.add_sync_job(sample_sync_job)

        sample_sync_job.status = SyncJobStatus.COMPLETED
        sample_sync_job.completed_at = datetime.now(UTC)
        result = temp_storage.update_sync_job(sample_sync_job)

        assert result.status == SyncJobStatus.COMPLETED
        assert result.completed_at is not None

    def test_update_sync_job_not_found_adds(self, temp_storage, sample_sync_job):
        """Test updating non-existent job adds it."""
        result = temp_storage.update_sync_job(sample_sync_job)
        assert result.id == sample_sync_job.id

        jobs = temp_storage.get_sync_history()
        assert len(jobs) == 1

    def test_get_last_sync_job(self, temp_storage):
        """Test getting most recent sync job."""
        source_id = str(uuid4())

        job1 = SyncJob(
            source_id=source_id,
            status=SyncJobStatus.COMPLETED,
            started_at=datetime(2024, 1, 1, tzinfo=UTC),
        )
        job2 = SyncJob(
            source_id=source_id,
            status=SyncJobStatus.COMPLETED,
            started_at=datetime(2024, 1, 2, tzinfo=UTC),
        )

        temp_storage.add_sync_job(job1)
        temp_storage.add_sync_job(job2)

        last = temp_storage.get_last_sync_job()
        assert last.id == job2.id

    def test_get_last_sync_job_filter_by_source(self, temp_storage):
        """Test getting last sync job filtered by source."""
        source1_id = str(uuid4())
        source2_id = str(uuid4())

        job1 = SyncJob(
            source_id=source1_id,
            status=SyncJobStatus.COMPLETED,
            started_at=datetime(2024, 1, 1, tzinfo=UTC),
        )
        job2 = SyncJob(
            source_id=source2_id,
            status=SyncJobStatus.COMPLETED,
            started_at=datetime(2024, 1, 2, tzinfo=UTC),
        )

        temp_storage.add_sync_job(job1)
        temp_storage.add_sync_job(job2)

        last = temp_storage.get_last_sync_job(source_id=source1_id)
        assert last.id == job1.id

    def test_get_last_sync_job_empty(self, temp_storage):
        """Test getting last sync job when none exist."""
        last = temp_storage.get_last_sync_job()
        assert last is None

    # ==================== Utilities ====================

    def test_load_json_file_not_exists(self, temp_storage):
        """Test _load_json returns default when file doesn't exist."""
        result = temp_storage._load_json(
            temp_storage.data_dir / "nonexistent.json",
            {"default": "value"},
        )
        assert result == {"default": "value"}

    def test_load_json_invalid_json(self, temp_storage):
        """Test _load_json returns default for invalid JSON."""
        invalid_file = temp_storage.data_dir / "invalid.json"
        invalid_file.write_text("not valid json{")

        result = temp_storage._load_json(invalid_file, {"default": "value"})
        assert result == {"default": "value"}

    def test_save_and_load_json(self, temp_storage):
        """Test _save_json and _load_json roundtrip."""
        test_file = temp_storage.data_dir / "test.json"
        data = {"key": "value", "number": 42, "korean": "한글"}

        temp_storage._save_json(test_file, data)
        loaded = temp_storage._load_json(test_file, {})

        assert loaded == data


class TestGetStorage:
    """Tests for get_storage function."""

    def test_get_storage_with_path(self, tmp_path):
        """Test get_storage with custom path."""
        storage = get_storage(tmp_path)
        assert storage.data_dir == tmp_path

    def test_get_storage_default_cached(self):
        """Test get_storage returns cached instance for default."""
        import src.storage

        # Reset global
        src.storage._storage = None

        storage1 = get_storage()
        storage2 = get_storage()
        assert storage1 is storage2

    def test_get_storage_custom_path_not_cached(self, tmp_path):
        """Test get_storage with custom path returns new instance."""
        storage1 = get_storage(tmp_path / "dir1")
        storage2 = get_storage(tmp_path / "dir2")
        assert storage1 is not storage2
