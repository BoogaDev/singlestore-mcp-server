"""
Test suite for SingleStore MCP Server
Run with: pytest tests/test_server.py -v
"""

import pytest
import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

# Set test environment variables before imports
os.environ["SINGLESTORE_HOST"] = "test.singlestore.com"
os.environ["SINGLESTORE_USER"] = "test_user"
os.environ["SINGLESTORE_PASSWORD"] = "test_password"
os.environ["SINGLESTORE_DATABASE"] = "test_db"
os.environ["MCP_API_KEY"] = "test_api_key"

from singlestore_mcp.server import SingleStoreMCPServer
from singlestore_mcp.database import SingleStoreManager, SingleStoreAdapter
from singlestore_mcp.tools import SingleStoreTools
from singlestore_mcp.security import SecurityManager


class TestSingleStoreManager:
    """Test database manager functionality"""

    @pytest.fixture
    async def db_manager(self):
        """Create a database manager instance"""
        manager = SingleStoreManager()
        await manager.initialize()
        return manager

    @pytest.mark.asyncio
    async def test_initialize_single_database(self):
        """Test initialization with single database configuration"""
        manager = SingleStoreManager()
        await manager.initialize()

        assert len(manager.databases) == 1
        assert "default" in manager.databases
        assert manager.databases["default"]["host"] == "test.singlestore.com"

    @pytest.mark.asyncio
    async def test_initialize_multiple_databases(self):
        """Test initialization with multiple database configurations"""
        configs = [
            {
                "id": "primary",
                "host": "primary.singlestore.com",
                "port": 3306,
                "user": "user1",
                "password": "pass1",
                "database": "db1",
            },
            {
                "id": "analytics",
                "host": "analytics.singlestore.com",
                "port": 3306,
                "user": "user2",
                "password": "pass2",
                "database": "db2",
            },
        ]

        os.environ["SINGLESTORE_CONFIGS"] = json.dumps(configs)

        manager = SingleStoreManager()
        await manager.initialize()

        assert len(manager.databases) == 2
        assert "primary" in manager.databases
        assert "analytics" in manager.databases

        # Clean up
        del os.environ["SINGLESTORE_CONFIGS"]

    @pytest.mark.asyncio
    async def test_connection_with_invalid_database_id(self, db_manager):
        """Test connection with invalid database ID"""
        with pytest.raises(ValueError, match="Database invalid_id not found"):
            async with db_manager.get_connection("invalid_id"):
                pass


class TestSingleStoreAdapter:
    """Test SingleStore adapter functionality"""

    @pytest.fixture
    def mock_connection(self):
        """Create a mock database connection"""
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_conn.cursor.return_value.__aenter__.return_value = mock_cursor
        return mock_conn, mock_cursor

    @pytest.mark.asyncio
    async def test_safe_query_validation(self):
        """Test SQL query safety validation"""
        adapter = SingleStoreAdapter(None)

        # Safe queries
        assert adapter._is_safe_query("SELECT * FROM users")
        assert adapter._is_safe_query("SHOW TABLES")
        assert adapter._is_safe_query("DESCRIBE users")
        assert adapter._is_safe_query("EXPLAIN SELECT * FROM users")

        # Unsafe queries
        assert not adapter._is_safe_query("INSERT INTO users VALUES (1, 'test')")
        assert not adapter._is_safe_query("UPDATE users SET name = 'test'")
        assert not adapter._is_safe_query("DELETE FROM users")
        assert not adapter._is_safe_query("DROP TABLE users")
        assert not adapter._is_safe_query("CREATE TABLE test (id INT)")

    @pytest.mark.asyncio
    async def test_execute_with_timeout(self, mock_connection):
        """Test query execution with timeout"""
        mock_conn, mock_cursor = mock_connection
        adapter = SingleStoreAdapter(mock_conn)

        mock_cursor.execute = AsyncMock()
        mock_cursor.fetchall = AsyncMock(return_value=[{"id": 1, "name": "test"}])

        result = await adapter.execute("SELECT * FROM users", timeout=10)

        # Verify timeout was set
        mock_cursor.execute.assert_any_call("SET SESSION max_execution_time = 10000")

        # Verify transaction handling
        mock_cursor.execute.assert_any_call("START TRANSACTION READ ONLY")
        mock_cursor.execute.assert_any_call("COMMIT")

        assert result == [{"id": 1, "name": "test"}]

    @pytest.mark.asyncio
    async def test_vector_search(self, mock_connection):
        """Test vector similarity search"""
        mock_conn, mock_cursor = mock_connection
        adapter = SingleStoreAdapter(mock_conn)

        mock_cursor.execute = AsyncMock()
        mock_cursor.fetchall = AsyncMock(
            return_value=[
                {"id": 1, "similarity_score": 0.95},
                {"id": 2, "similarity_score": 0.87},
            ]
        )

        query_vector = [0.1, 0.2, 0.3, 0.4]
        result = await adapter.vector_search(
            "products", "embedding", query_vector, limit=5
        )

        # Verify the vector search query was constructed correctly
        call_args = mock_cursor.execute.call_args_list
        vector_query = None
        for call in call_args:
            if "DOT_PRODUCT" in str(call):
                vector_query = call[0][0]
                break

        assert vector_query is not None
        assert "DOT_PRODUCT" in vector_query
        assert "ORDER BY similarity_score DESC" in vector_query
        assert "LIMIT 5" in vector_query


class TestSingleStoreTools:
    """Test database tools functionality"""

    @pytest.fixture
    async def tools(self):
        """Create tools instance with mock manager"""
        manager = MagicMock(spec=SingleStoreManager)
        return SingleStoreTools(manager)

    @pytest.mark.asyncio
    async def test_execute_query_success(self, tools):
        """Test successful query execution"""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(
            return_value=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        )

        tools.db_manager.get_connection = AsyncMock()
        tools.db_manager.get_connection.return_value.__aenter__.return_value = mock_conn

        result = await tools.execute_query("SELECT * FROM users")

        assert result["success"] is True
        assert result["row_count"] == 2
        assert len(result["data"]) == 2
        assert result["truncated"] is False

    @pytest.mark.asyncio
    async def test_execute_query_with_truncation(self, tools):
        """Test query result truncation for large datasets"""
        # Create 1500 mock results
        mock_data = [{"id": i, "value": f"data_{i}"} for i in range(1500)]

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value=mock_data)

        tools.db_manager.get_connection = AsyncMock()
        tools.db_manager.get_connection.return_value.__aenter__.return_value = mock_conn

        result = await tools.execute_query("SELECT * FROM large_table")

        assert result["success"] is True
        assert result["row_count"] == 1000  # Truncated to 1000
        assert len(result["data"]) == 1000
        assert result["truncated"] is True

    @pytest.mark.asyncio
    async def test_execute_query_error(self, tools):
        """Test query execution error handling"""
        tools.db_manager.get_connection = AsyncMock()
        tools.db_manager.get_connection.side_effect = Exception("Connection failed")

        result = await tools.execute_query("SELECT * FROM users")

        assert result["success"] is False
        assert "Connection failed" in result["error"]

    @pytest.mark.asyncio
    async def test_vector_search_success(self, tools):
        """Test successful vector search"""
        mock_conn = AsyncMock()
        mock_conn.vector_search = AsyncMock(
            return_value=[
                {"id": 1, "similarity_score": 0.95},
                {"id": 2, "similarity_score": 0.87},
            ]
        )

        tools.db_manager.get_connection = AsyncMock()
        tools.db_manager.get_connection.return_value.__aenter__.return_value = mock_conn

        result = await tools.vector_search(
            "products", "embedding", [0.1, 0.2, 0.3], limit=5
        )

        assert result["success"] is True
        assert result["count"] == 2
        assert len(result["results"]) == 2


class TestSecurityManager:
    """Test security functionality"""

    @pytest.fixture
    def security(self):
        """Create security manager instance"""
        return SecurityManager()

    @pytest.mark.asyncio
    async def test_validate_auth_with_valid_token(self, security):
        """Test authentication with valid token"""
        auth_header = "Bearer test_api_key"
        result = await security.validate_auth(auth_header)
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_auth_with_invalid_token(self, security):
        """Test authentication with invalid token"""
        auth_header = "Bearer wrong_key"
        result = await security.validate_auth(auth_header)
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_auth_without_bearer(self, security):
        """Test authentication without Bearer prefix"""
        auth_header = "test_api_key"
        result = await security.validate_auth(auth_header)
        assert result is False

    @pytest.mark.asyncio
    async def test_rate_limiting(self, security):
        """Test rate limiting functionality"""
        client_id = "test_client"

        # Should allow initial requests
        for i in range(10):
            assert security._check_rate_limit(client_id) is True

        # Set up to exceed rate limit
        security.max_requests_per_minute = 10

        # This should fail
        assert security._check_rate_limit(client_id) is False

    def test_sql_validation(self, security):
        """Test SQL injection prevention"""
        # Safe queries
        assert security._validate_sql("SELECT * FROM users") is True
        assert (
            security._validate_sql("SELECT id, name FROM products WHERE price > 100")
            is True
        )

        # Dangerous queries
        assert (
            security._validate_sql("SELECT * FROM users; DROP TABLE users;--") is False
        )
        assert (
            security._validate_sql("SELECT * FROM users WHERE id = 1 OR 1=1") is False
        )
        assert (
            security._validate_sql("SELECT * FROM users UNION SELECT * FROM passwords")
            is False
        )

        # Query too long
        long_query = (
            "SELECT " + ", ".join([f"col_{i}" for i in range(5000)]) + " FROM table"
        )
        assert security._validate_sql(long_query) is False


class TestSingleStoreMCPServer:
    """Test MCP server functionality"""

    @pytest.fixture
    async def server(self):
        """Create server instance"""
        server = SingleStoreMCPServer()
        await server.db_manager.initialize()
        return server

    @pytest.mark.asyncio
    async def test_list_tools(self, server):
        """Test listing available tools"""
        # Mock the list_tools handler
        tools = await server.server.list_tools()

        tool_names = [tool.name for tool in tools]

        assert "query_database" in tool_names
        assert "vector_search" in tool_names
        assert "list_tables" in tool_names
        assert "describe_table" in tool_names
        assert "analyze_query" in tool_names
        assert "get_table_statistics" in tool_names

    @pytest.mark.asyncio
    async def test_call_unknown_tool(self, server):
        """Test calling an unknown tool"""
        result = await server.server.call_tool("unknown_tool", {})

        assert len(result) == 1
        assert "Error" in result[0].text or "Unknown tool" in result[0].text


class TestIntegration:
    """Integration tests"""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("INTEGRATION_TEST"),
        reason="Integration tests require real database connection",
    )
    async def test_real_database_connection(self):
        """Test with real SingleStore database connection"""
        # This test requires a real SingleStore instance
        # Set INTEGRATION_TEST=1 and configure real credentials to run

        manager = SingleStoreManager()
        await manager.initialize()

        async with manager.get_connection() as conn:
            # Test basic query
            result = await conn.execute("SELECT 1 as test")
            assert result[0]["test"] == 1

            # Test listing tables
            tables = await conn.get_tables()
            assert isinstance(tables, list)

    @pytest.mark.asyncio
    async def test_end_to_end_query_flow(self):
        """Test complete query flow from server to database"""
        server = SingleStoreMCPServer()

        # Mock the database connection
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value=[{"id": 1, "name": "Test User"}])

        server.db_manager.get_connection = AsyncMock()
        server.db_manager.get_connection.return_value.__aenter__.return_value = (
            mock_conn
        )

        # Call the query tool
        result = await server.server.call_tool(
            "query_database", {"sql": "SELECT * FROM users LIMIT 1"}
        )

        # Verify result
        assert len(result) == 1
        result_data = json.loads(result[0].text)
        assert result_data["success"] is True
        assert result_data["row_count"] == 1


# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line("markers", "integration: mark test as integration test")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--cov=singlestore_mcp", "--cov-report=term-missing"])
