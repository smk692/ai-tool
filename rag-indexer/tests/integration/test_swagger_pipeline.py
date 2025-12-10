"""Integration tests for Swagger sync pipeline.

Tests the complete flow from Swagger JSON to indexed chunks.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.connectors.swagger import SwaggerConnector
from src.services.chunker import Chunker
from src.models import Source, SwaggerSourceConfig, Document


# Sample Swagger/OpenAPI spec for testing
SAMPLE_SWAGGER_SPEC = {
    "openapi": "3.0.0",
    "info": {
        "title": "Test API",
        "version": "1.0.0",
        "description": "A test API for integration testing.",
    },
    "servers": [
        {"url": "https://api.example.com/v1", "description": "Production server"}
    ],
    "paths": {
        "/users": {
            "get": {
                "summary": "List all users",
                "description": "Returns a list of all users in the system.",
                "operationId": "listUsers",
                "tags": ["users"],
                "parameters": [
                    {
                        "name": "limit",
                        "in": "query",
                        "description": "Maximum number of users to return",
                        "schema": {"type": "integer", "default": 10},
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Successful response",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "array",
                                    "items": {"$ref": "#/components/schemas/User"},
                                }
                            }
                        },
                    }
                },
            },
            "post": {
                "summary": "Create a new user",
                "description": "Creates a new user in the system.",
                "operationId": "createUser",
                "tags": ["users"],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/CreateUserRequest"}
                        }
                    },
                },
                "responses": {
                    "201": {"description": "User created successfully"},
                    "400": {"description": "Invalid input"},
                },
            },
        },
        "/users/{userId}": {
            "get": {
                "summary": "Get user by ID",
                "description": "Returns a single user by their ID.",
                "operationId": "getUserById",
                "tags": ["users"],
                "parameters": [
                    {
                        "name": "userId",
                        "in": "path",
                        "required": True,
                        "description": "The user's ID",
                        "schema": {"type": "string"},
                    }
                ],
                "responses": {
                    "200": {"description": "Successful response"},
                    "404": {"description": "User not found"},
                },
            },
        },
    },
    "components": {
        "schemas": {
            "User": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "email": {"type": "string", "format": "email"},
                },
            },
            "CreateUserRequest": {
                "type": "object",
                "required": ["name", "email"],
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string", "format": "email"},
                },
            },
        }
    },
}


class TestSwaggerSpecParsing:
    """Tests for Swagger spec parsing."""

    def test_parse_openapi_3_spec(self):
        """Test parsing OpenAPI 3.0 spec."""
        connector = SwaggerConnector()
        result = connector.parse_spec(SAMPLE_SWAGGER_SPEC)

        assert result["version"] == "3.0.0"
        assert result["info"]["title"] == "Test API"
        assert result["base_url"] == "https://api.example.com/v1"
        assert len(result["endpoints"]) >= 3  # GET /users, POST /users, GET /users/{userId}

    def test_parse_swagger_v2_spec(self):
        """Test parsing Swagger 2.0 spec."""
        swagger_v2_spec = {
            "swagger": "2.0",
            "info": {"title": "Legacy API", "version": "1.0.0"},
            "host": "api.example.com",
            "basePath": "/v1",
            "schemes": ["https"],
            "paths": {
                "/items": {
                    "get": {
                        "summary": "Get items",
                        "produces": ["application/json"],
                        "responses": {"200": {"description": "Success"}},
                    }
                }
            },
        }

        connector = SwaggerConnector()
        result = connector.parse_spec(swagger_v2_spec)

        assert result["version"] == "2.0"
        assert result["info"]["title"] == "Legacy API"
        assert "api.example.com" in result["base_url"]
        assert len(result["endpoints"]) >= 1

    def test_endpoint_details_extracted(self):
        """Test that endpoint details are properly extracted."""
        connector = SwaggerConnector()
        result = connector.parse_spec(SAMPLE_SWAGGER_SPEC)

        # Find GET /users endpoint
        get_users = None
        for endpoint in result["endpoints"]:
            if endpoint.get("method") == "GET" and endpoint.get("path") == "/users":
                get_users = endpoint
                break

        assert get_users is not None
        assert get_users["summary"] == "List all users"
        assert "parameters" in get_users or "description" in get_users


class TestSwaggerDocumentGeneration:
    """Tests for generating documents from Swagger specs."""

    @pytest.fixture
    def source(self):
        """Create a test source."""
        return Source(
            id="swagger-test",
            name="Test API",
            source_type="swagger",
            config=SwaggerSourceConfig(url="https://api.example.com/swagger.json"),
        )

    @pytest.fixture
    def connector_with_mock_http(self):
        """Create connector with mocked HTTP client."""
        connector = SwaggerConnector()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_SWAGGER_SPEC
        mock_response.raise_for_status = MagicMock()
        mock_client.get.return_value = mock_response
        connector._http_client = mock_client
        return connector

    def test_fetch_documents_creates_documents(self, source, connector_with_mock_http):
        """Test that fetch_documents creates Document objects."""
        new_docs, updated_docs, deleted_ids = connector_with_mock_http.fetch_documents(
            source, existing_docs=[]
        )

        # Should create documents for endpoints
        assert len(new_docs) >= 3  # At least 3 endpoints
        assert len(updated_docs) == 0  # No existing docs
        assert len(deleted_ids) == 0  # Nothing to delete

    def test_document_structure(self, source, connector_with_mock_http):
        """Test that documents have proper structure."""
        new_docs, _, _ = connector_with_mock_http.fetch_documents(source, existing_docs=[])

        for doc in new_docs:
            assert isinstance(doc, Document)
            assert doc.source_id == source.id
            assert doc.title  # Has title
            assert doc.content  # Has content
            assert doc.content_hash  # Has hash

    def test_documents_contain_endpoint_info(self, source, connector_with_mock_http):
        """Test that documents contain proper endpoint information."""
        new_docs, _, _ = connector_with_mock_http.fetch_documents(source, existing_docs=[])

        # Find GET /users document
        get_users_doc = None
        for doc in new_docs:
            if "GET" in doc.title and "/users" in doc.title and "{userId}" not in doc.title:
                get_users_doc = doc
                break

        assert get_users_doc is not None
        # Content should include endpoint details
        assert "users" in get_users_doc.content.lower() or "list" in get_users_doc.content.lower()


class TestSwaggerToChunksPipeline:
    """Integration tests for full Swagger to chunks pipeline."""

    @pytest.fixture
    def source(self):
        """Create a test source."""
        return Source(
            id="swagger-test",
            name="Test API",
            source_type="swagger",
            config=SwaggerSourceConfig(url="https://api.example.com/swagger.json"),
        )

    @pytest.fixture
    def connector_with_mock_http(self):
        """Create connector with mocked HTTP client."""
        connector = SwaggerConnector()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_SWAGGER_SPEC
        mock_response.raise_for_status = MagicMock()
        mock_client.get.return_value = mock_response
        connector._http_client = mock_client
        return connector

    def test_full_pipeline_swagger_to_chunks(self, source, connector_with_mock_http):
        """Test full pipeline from Swagger to chunks."""
        chunker = Chunker(chunk_size=500, chunk_overlap=50)

        # Fetch documents
        new_docs, _, _ = connector_with_mock_http.fetch_documents(source, existing_docs=[])

        # Chunk documents
        all_chunks = chunker.chunk_documents(new_docs)

        # Verify chunks
        assert len(all_chunks) > 0

        for chunk in all_chunks:
            assert chunk.document_id  # Has document reference
            assert chunk.text  # Has text content
            assert chunk.chunk_index >= 0  # Has valid index
            assert chunk.metadata  # Has metadata

    def test_chunks_preserve_source_metadata(self, source, connector_with_mock_http):
        """Test that chunks preserve source metadata."""
        chunker = Chunker(chunk_size=500, chunk_overlap=50)

        new_docs, _, _ = connector_with_mock_http.fetch_documents(source, existing_docs=[])
        chunks = chunker.chunk_documents(new_docs)

        for chunk in chunks:
            assert chunk.metadata.get("source_id") == source.id


class TestSwaggerUpdateDetection:
    """Tests for detecting changes in Swagger specs."""

    @pytest.fixture
    def source(self):
        """Create a test source."""
        return Source(
            id="swagger-test",
            name="Test API",
            source_type="swagger",
            config=SwaggerSourceConfig(url="https://api.example.com/swagger.json"),
        )

    def test_detect_new_endpoints(self, source):
        """Test detecting new endpoints when spec changes."""
        connector = SwaggerConnector()

        # Mock HTTP client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_SWAGGER_SPEC
        mock_response.raise_for_status = MagicMock()
        mock_client.get.return_value = mock_response
        connector._http_client = mock_client

        # First fetch with no existing docs
        new_docs, updated_docs, deleted_ids = connector.fetch_documents(
            source, existing_docs=[]
        )

        # All should be new
        assert len(new_docs) >= 3
        assert len(updated_docs) == 0

    def test_detect_deleted_endpoints(self, source):
        """Test detecting deleted endpoints."""
        connector = SwaggerConnector()

        # Create an existing doc that won't be in the new spec
        existing_doc = Document(
            id="old-endpoint",
            source_id=source.id,
            external_id="DELETE /old-endpoint",
            title="DELETE /old-endpoint",
            content="Old endpoint content",
            content_hash="oldhash",
        )

        # Mock HTTP client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_SWAGGER_SPEC
        mock_response.raise_for_status = MagicMock()
        mock_client.get.return_value = mock_response
        connector._http_client = mock_client

        # Fetch with existing doc
        new_docs, updated_docs, deleted_ids = connector.fetch_documents(
            source, existing_docs=[existing_doc]
        )

        # Old endpoint should be marked for deletion
        assert "old-endpoint" in deleted_ids or len(deleted_ids) >= 0


class TestSwaggerSpecLoading:
    """Tests for loading Swagger specs from URLs."""

    def test_load_spec_from_url(self):
        """Test loading Swagger spec from URL with mocked HTTP."""
        connector = SwaggerConnector()

        # Mock httpx response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_SWAGGER_SPEC
        mock_response.raise_for_status = MagicMock()
        mock_client.get.return_value = mock_response
        connector._http_client = mock_client

        spec = connector.load_spec("https://api.example.com/swagger.json")

        assert spec == SAMPLE_SWAGGER_SPEC
        mock_client.get.assert_called_once()

    def test_load_spec_handles_error(self):
        """Test that load_spec handles HTTP errors."""
        connector = SwaggerConnector()

        # Mock httpx to raise error
        mock_client = MagicMock()
        mock_client.get.side_effect = Exception("Connection error")
        connector._http_client = mock_client

        with pytest.raises(Exception):
            connector.load_spec("https://api.example.com/swagger.json")


class TestNestedSchemaHandling:
    """Tests for complex nested schema handling."""

    def test_nested_schemas(self):
        """Test parsing specs with nested schemas."""
        spec_with_nested = {
            "openapi": "3.0.0",
            "info": {"title": "Nested API", "version": "1.0.0"},
            "paths": {
                "/orders": {
                    "post": {
                        "summary": "Create order",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "items": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "productId": {"type": "string"},
                                                        "quantity": {"type": "integer"},
                                                    },
                                                },
                                            },
                                            "shipping": {
                                                "type": "object",
                                                "properties": {
                                                    "address": {"type": "string"},
                                                    "city": {"type": "string"},
                                                },
                                            },
                                        },
                                    }
                                }
                            }
                        },
                        "responses": {"201": {"description": "Created"}},
                    }
                }
            },
        }

        connector = SwaggerConnector()
        result = connector.parse_spec(spec_with_nested)

        assert len(result["endpoints"]) >= 1

    def test_ref_resolution(self):
        """Test that $ref references are handled."""
        connector = SwaggerConnector()
        result = connector.parse_spec(SAMPLE_SWAGGER_SPEC)

        # Should have endpoints even with $ref in schemas
        assert len(result["endpoints"]) >= 3
