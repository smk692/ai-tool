"""Integration tests for CLI commands.

Tests CLI commands with mocked external services.
"""

import pytest
from unittest.mock import MagicMock, patch
from typer.testing import CliRunner

from src.cli import app


class TestCLIHelp:
    """Tests for CLI help commands."""

    @pytest.fixture
    def cli_runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_cli_help(self, cli_runner):
        """Test CLI help command."""
        result = cli_runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "rag-indexer" in result.output.lower() or "RAG" in result.output

    def test_source_help(self, cli_runner):
        """Test source command help."""
        result = cli_runner.invoke(app, ["source", "--help"])

        assert result.exit_code == 0
        assert "source" in result.output.lower()

    def test_sync_help(self, cli_runner):
        """Test sync command help."""
        result = cli_runner.invoke(app, ["sync", "--help"])

        assert result.exit_code == 0
        assert "sync" in result.output.lower()

    def test_scheduler_help(self, cli_runner):
        """Test scheduler command help."""
        result = cli_runner.invoke(app, ["scheduler", "--help"])

        assert result.exit_code == 0
        assert "scheduler" in result.output.lower()

    def test_search_help(self, cli_runner):
        """Test search command help."""
        result = cli_runner.invoke(app, ["search", "--help"])

        assert result.exit_code == 0
        assert "search" in result.output.lower()

    def test_status_help(self, cli_runner):
        """Test status command help."""
        result = cli_runner.invoke(app, ["status", "--help"])

        assert result.exit_code == 0


class TestSourceCommands:
    """Tests for source management commands."""

    @pytest.fixture
    def cli_runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def mock_storage(self):
        """Create mock storage."""
        with patch("src.cli.get_storage") as mock:
            mock_instance = MagicMock()
            mock_instance.get_sources.return_value = []
            mock_instance.get_source.return_value = None
            mock_instance.get_source_by_name.return_value = None
            mock.return_value = mock_instance
            yield mock_instance

    def test_source_list_empty(self, cli_runner, mock_storage):
        """Test source list with no sources."""
        result = cli_runner.invoke(app, ["source", "list"])

        assert result.exit_code == 0
        assert "No sources" in result.output or result.output != ""

    def test_source_add_notion_missing_args(self, cli_runner, mock_storage):
        """Test adding Notion source without required args."""
        result = cli_runner.invoke(
            app,
            ["source", "add", "--name", "Test", "--type", "notion"],
        )

        # Should fail - missing page-ids or database-ids
        assert result.exit_code == 1

    def test_source_add_swagger_missing_url(self, cli_runner, mock_storage):
        """Test adding Swagger source without URL."""
        result = cli_runner.invoke(
            app,
            ["source", "add", "--name", "Test", "--type", "swagger"],
        )

        # Should fail - missing url
        assert result.exit_code == 1

    def test_source_add_swagger_success(self, cli_runner, mock_storage):
        """Test adding Swagger source successfully."""
        result = cli_runner.invoke(
            app,
            [
                "source", "add",
                "--name", "Test API",
                "--type", "swagger",
                "--url", "https://api.example.com/swagger.json",
            ],
        )

        assert result.exit_code == 0
        assert "added successfully" in result.output.lower()

    def test_source_add_notion_success(self, cli_runner, mock_storage):
        """Test adding Notion source successfully."""
        result = cli_runner.invoke(
            app,
            [
                "source", "add",
                "--name", "Test Notion",
                "--type", "notion",
                "--page-ids", "page-1,page-2",
            ],
        )

        assert result.exit_code == 0
        assert "added successfully" in result.output.lower()

    def test_source_remove_not_found(self, cli_runner, mock_storage):
        """Test removing non-existent source."""
        result = cli_runner.invoke(app, ["source", "remove", "nonexistent"])

        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_source_show_not_found(self, cli_runner, mock_storage):
        """Test showing non-existent source."""
        result = cli_runner.invoke(app, ["source", "show", "nonexistent"])

        assert result.exit_code == 1
        assert "not found" in result.output.lower()


class TestSyncCommands:
    """Tests for sync commands."""

    @pytest.fixture
    def cli_runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def mock_scheduler(self):
        """Create mock scheduler."""
        with patch("src.cli.get_scheduler") as mock:
            mock_instance = MagicMock()

            # Create a mock job with all required attributes
            mock_job = MagicMock()
            mock_job.status.value = "completed"
            mock_job.documents_processed = 10
            mock_job.documents_created = 5
            mock_job.documents_updated = 3
            mock_job.documents_deleted = 2
            mock_job.chunks_created = 50
            mock_job.errors = []
            mock_job.duration_seconds = 1.5
            mock_job.error_message = None

            mock_instance.trigger_sync.return_value = mock_job
            mock.return_value = mock_instance
            yield mock_instance

    def test_sync_run(self, cli_runner, mock_scheduler):
        """Test sync run command."""
        result = cli_runner.invoke(app, ["sync", "run"])

        assert result.exit_code == 0
        assert "Sync" in result.output

    @pytest.fixture
    def mock_storage_for_history(self):
        """Create mock storage for sync history."""
        with patch("src.cli.get_storage") as mock:
            mock_instance = MagicMock()
            mock_instance.get_sync_history.return_value = []
            mock.return_value = mock_instance
            yield mock_instance

    def test_sync_history_empty(self, cli_runner, mock_storage_for_history):
        """Test sync history with no jobs."""
        result = cli_runner.invoke(app, ["sync", "history"])

        assert result.exit_code == 0
        assert "No sync history" in result.output or result.output != ""


class TestSchedulerCommands:
    """Tests for scheduler commands."""

    @pytest.fixture
    def cli_runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def mock_scheduler(self):
        """Create mock scheduler."""
        with patch("src.cli.get_scheduler") as mock:
            mock_instance = MagicMock()
            mock_instance.is_running.return_value = False
            mock_instance.cron_expression = "0 */6 * * *"
            mock_instance.get_next_run.return_value = None
            mock.return_value = mock_instance
            yield mock_instance

    def test_scheduler_start(self, cli_runner, mock_scheduler):
        """Test scheduler start command."""
        result = cli_runner.invoke(app, ["scheduler", "start"])

        assert result.exit_code == 0
        assert "started" in result.output.lower()

    def test_scheduler_stop_not_running(self, cli_runner, mock_scheduler):
        """Test scheduler stop when not running."""
        result = cli_runner.invoke(app, ["scheduler", "stop"])

        assert result.exit_code == 0
        assert "not running" in result.output.lower()

    def test_scheduler_status_not_running(self, cli_runner, mock_scheduler):
        """Test scheduler status when not running."""
        result = cli_runner.invoke(app, ["scheduler", "status"])

        assert result.exit_code == 0
        assert "not running" in result.output.lower()


class TestStatusCommand:
    """Tests for status command."""

    @pytest.fixture
    def cli_runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def mock_all_services(self):
        """Create mocks for all services used by status."""
        with patch("src.cli.get_storage") as mock_storage, \
             patch("src.cli.get_indexer") as mock_indexer, \
             patch("src.cli.get_scheduler") as mock_scheduler:

            # Storage mock
            storage_instance = MagicMock()
            storage_instance.get_sources.return_value = []
            storage_instance.get_documents.return_value = []
            storage_instance.get_last_sync_job.return_value = None
            mock_storage.return_value = storage_instance

            # Indexer mock
            indexer_instance = MagicMock()
            indexer_instance.collection_name = "test-collection"
            indexer_instance.get_collection_stats.return_value = {"total_chunks": 100}
            mock_indexer.return_value = indexer_instance

            # Scheduler mock
            scheduler_instance = MagicMock()
            scheduler_instance.is_running.return_value = False
            scheduler_instance.get_next_run.return_value = None
            mock_scheduler.return_value = scheduler_instance

            yield {
                "storage": storage_instance,
                "indexer": indexer_instance,
                "scheduler": scheduler_instance,
            }

    def test_status_command(self, cli_runner, mock_all_services):
        """Test status command shows system status."""
        result = cli_runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "Status" in result.output or "Sources" in result.output


class TestSearchCommand:
    """Tests for search command."""

    @pytest.fixture
    def cli_runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def mock_indexer(self):
        """Create mock indexer for search."""
        with patch("src.cli.get_indexer") as mock:
            mock_instance = MagicMock()
            mock_instance.search.return_value = []
            mock.return_value = mock_instance
            yield mock_instance

    def test_search_no_results(self, cli_runner, mock_indexer):
        """Test search with no results."""
        result = cli_runner.invoke(app, ["search", "test query"])

        assert result.exit_code == 0
        assert "No results" in result.output

    def test_search_with_results(self, cli_runner, mock_indexer):
        """Test search with results."""
        mock_indexer.search.return_value = [
            {
                "score": 0.95,
                "payload": {
                    "text": "Test result content",
                    "source_type": "notion",
                    "title": "Test Document",
                    "url": "https://example.com",
                },
            }
        ]

        result = cli_runner.invoke(app, ["search", "test query"])

        assert result.exit_code == 0
        assert "Found" in result.output or "Result" in result.output


class TestCLIErrorHandling:
    """Tests for CLI error handling."""

    @pytest.fixture
    def cli_runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_invalid_command(self, cli_runner):
        """Test CLI handles invalid command gracefully."""
        result = cli_runner.invoke(app, ["invalid-command"])

        # Should report error, not crash
        assert result.exit_code != 0

    def test_missing_required_args(self, cli_runner):
        """Test CLI handles missing required arguments."""
        result = cli_runner.invoke(app, ["search"])  # Missing query

        # Should report missing argument
        assert result.exit_code != 0


class TestCLIOptions:
    """Tests for CLI global options."""

    @pytest.fixture
    def cli_runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def mock_all_services(self):
        """Create mocks for all services."""
        with patch("src.cli.get_storage") as mock_storage, \
             patch("src.cli.get_indexer") as mock_indexer, \
             patch("src.cli.get_scheduler") as mock_scheduler:

            storage_instance = MagicMock()
            storage_instance.get_sources.return_value = []
            storage_instance.get_documents.return_value = []
            storage_instance.get_last_sync_job.return_value = None
            mock_storage.return_value = storage_instance

            indexer_instance = MagicMock()
            indexer_instance.collection_name = "test"
            indexer_instance.get_collection_stats.return_value = {"total_chunks": 0}
            mock_indexer.return_value = indexer_instance

            scheduler_instance = MagicMock()
            scheduler_instance.is_running.return_value = False
            scheduler_instance.get_next_run.return_value = None
            mock_scheduler.return_value = scheduler_instance

            yield

    def test_verbose_mode(self, cli_runner, mock_all_services):
        """Test verbose mode option."""
        result = cli_runner.invoke(app, ["--verbose", "status"])

        # Should execute with verbose flag
        assert result.exit_code == 0
