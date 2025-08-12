import os
import json
import logging
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager
import singlestoredb as s2
import aiomysql
import sqlparse
from asyncio_throttle import Throttler

logger = logging.getLogger(__name__)


class SingleStoreManager:
    def __init__(self):
        self.databases = {}
        self.throttler = Throttler(rate_limit=100, period=60)  # 100 queries per minute

    async def initialize(self):
        """Initialize database connections from environment"""
        # Support multiple database configurations
        db_configs = os.getenv("SINGLESTORE_CONFIGS")
        if db_configs:
            configs = json.loads(db_configs)
            for config in configs:
                await self.add_database(config)
        # Support single database configuration
        elif os.getenv("SINGLESTORE_HOST"):
            await self.add_database(
                {
                    "id": "default",
                    "host": os.getenv("SINGLESTORE_HOST"),
                    "port": int(os.getenv("SINGLESTORE_PORT", 3306)),
                    "user": os.getenv("SINGLESTORE_USER"),
                    "password": os.getenv("SINGLESTORE_PASSWORD"),
                    "database": os.getenv("SINGLESTORE_DATABASE"),
                    "description": "Default SingleStore database",
                }
            )

    async def add_database(self, config: Dict[str, Any]):
        """Add a SingleStore database configuration"""
        db_id = config.get("id", f"singlestore_{len(self.databases)}")
        self.databases[db_id] = config
        logger.info(f"Added SingleStore database: {db_id}")

    @asynccontextmanager
    async def get_connection(
        self, database_id: Optional[str] = None, use_native: bool = True
    ):
        """Get a database connection with read-only transaction"""
        async with self.throttler:
            if not database_id and len(self.databases) == 1:
                database_id = list(self.databases.keys())[0]

            if database_id not in self.databases:
                raise ValueError(f"Database {database_id} not found")

            config = self.databases[database_id]

            if use_native:
                # Use native SingleStore client
                conn = await self._get_native_connection(config)
                yield SingleStoreAdapter(conn)
                await conn.aclose()
            else:
                # Use aiomysql for compatibility
                conn = await self._get_aiomysql_connection(config)
                yield MySQLCompatAdapter(conn)
                conn.close()

    async def _get_native_connection(self, config: Dict[str, Any]):
        """Create native SingleStore connection"""
        connection_params = {
            "host": config["host"],
            "port": config.get("port", 3306),
            "user": config["user"],
            "password": config["password"],
            "database": config["database"],
            "autocommit": False,  # Use transactions
            "results_type": "dict",  # Return results as dictionaries
        }

        # Add SSL if configured
        if os.getenv("SINGLESTORE_SSL_CERT"):
            connection_params["ssl"] = {
                "ca": os.getenv("SINGLESTORE_SSL_CERT"),
                "verify_cert": True,
            }

        return await s2.connect_async(**connection_params)

    async def _get_aiomysql_connection(self, config: Dict[str, Any]):
        """Create aiomysql connection for compatibility"""
        connection_params = {
            "host": config["host"],
            "port": config.get("port", 3306),
            "user": config["user"],
            "password": config["password"],
            "db": config["database"],
            "autocommit": False,
        }

        return await aiomysql.connect(**connection_params)


class SingleStoreAdapter:
    """Adapter for native SingleStore connections"""

    def __init__(self, connection):
        self.conn = connection

    async def execute(self, query: str, timeout: int = 30) -> List[Dict[str, Any]]:
        """Execute a query and return results as list of dicts"""
        # Validate query is safe
        if not self._is_safe_query(query):
            raise ValueError("Only SELECT queries and SHOW commands are allowed")

        async with self.conn.cursor() as cursor:
            # Set query timeout
            await cursor.execute(f"SET SESSION max_execution_time = {timeout * 1000}")

            # Execute in read-only transaction
            await cursor.execute("START TRANSACTION READ ONLY")
            try:
                await cursor.execute(query)
                results = await cursor.fetchall()
                await cursor.execute("COMMIT")
                return results if results else []
            except Exception as e:
                await cursor.execute("ROLLBACK")
                raise e

    async def vector_search(
        self, table: str, vector_column: str, query_vector: List[float], limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search using SingleStore's DOT_PRODUCT"""
        # Build the vector search query
        vector_str = f"JSON_ARRAY_PACK('[{','.join(map(str, query_vector))}]')"

        query = f"""
        SELECT *, 
               DOT_PRODUCT({vector_column}, {vector_str}) AS similarity_score
        FROM {table}
        ORDER BY similarity_score DESC
        LIMIT {limit}
        """

        return await self.execute(query)

    async def get_tables(self) -> List[str]:
        """Get list of tables"""
        query = "SHOW TABLES"
        results = await self.execute(query)
        return [list(row.values())[0] for row in results]

    async def describe_table(self, table_name: str) -> Dict[str, Any]:
        """Get table structure with SingleStore-specific features"""
        # Get columns
        columns_query = f"DESCRIBE {table_name}"
        columns = await self.execute(columns_query)

        # Get indexes
        indexes_query = f"SHOW INDEXES FROM {table_name}"
        indexes = await self.execute(indexes_query)

        # Get table status (row count, size, etc.)
        status_query = f"SHOW TABLE STATUS LIKE '{table_name}'"
        status = await self.execute(status_query)

        # Check if it's a columnstore table
        columnstore_query = f"""
        SELECT STORAGE_TYPE 
        FROM information_schema.TABLES 
        WHERE TABLE_NAME = '{table_name}'
        """
        storage_info = await self.execute(columnstore_query)

        return {
            "table_name": table_name,
            "columns": columns,
            "indexes": indexes,
            "status": status[0] if status else {},
            "storage_type": (
                storage_info[0]["STORAGE_TYPE"] if storage_info else "UNKNOWN"
            ),
        }

    async def get_query_profile(self, query: str) -> Dict[str, Any]:
        """Get detailed query execution profile"""
        async with self.conn.cursor() as cursor:
            # Enable profiling
            await cursor.execute("SET PROFILE = ON")
            await cursor.execute("SET PROFILE_LEVEL = FULL")

            # Execute the query
            await cursor.execute(query)

            # Get the profile
            await cursor.execute("SHOW PROFILE")
            profile = await cursor.fetchall()

            # Disable profiling
            await cursor.execute("SET PROFILE = OFF")

            return {"query": query, "profile": profile}

    def _is_safe_query(self, query: str) -> bool:
        """Check if query is safe (read-only)"""
        # Parse the SQL
        parsed = sqlparse.parse(query)
        if not parsed:
            return False

        # Check first statement type
        statement = parsed[0]
        stmt_type = statement.get_type()

        # Allow SELECT, SHOW, DESCRIBE, EXPLAIN
        allowed_types = ["SELECT", "SHOW", "DESCRIBE", "EXPLAIN", "UNKNOWN"]

        if stmt_type not in allowed_types:
            return False

        # Additional check for dangerous keywords
        unsafe_keywords = [
            "INSERT",
            "UPDATE",
            "DELETE",
            "DROP",
            "CREATE",
            "ALTER",
            "TRUNCATE",
            "GRANT",
            "REVOKE",
            "CALL",
        ]
        query_upper = query.upper()

        return not any(keyword in query_upper for keyword in unsafe_keywords)


class MySQLCompatAdapter:
    """Adapter for aiomysql connections (MySQL compatibility mode)"""

    def __init__(self, connection):
        self.conn = connection

    async def execute(self, query: str, timeout: int = 30) -> List[Dict[str, Any]]:
        """Execute a query and return results"""
        if not self._is_safe_query(query):
            raise ValueError("Only SELECT queries are allowed")

        async with self.conn.cursor(aiomysql.DictCursor) as cursor:
            # Set query timeout
            await cursor.execute(f"SET SESSION max_execution_time = {timeout * 1000}")

            # Execute query
            await cursor.execute(query)
            results = await cursor.fetchall()
            return results if results else []

    async def get_tables(self) -> List[str]:
        """Get list of tables"""
        async with self.conn.cursor() as cursor:
            await cursor.execute("SHOW TABLES")
            return [row[0] for row in await cursor.fetchall()]

    async def describe_table(self, table_name: str) -> Dict[str, Any]:
        """Get table structure"""
        async with self.conn.cursor(aiomysql.DictCursor) as cursor:
            await cursor.execute(f"DESCRIBE {table_name}")
            columns = await cursor.fetchall()

            return {"table_name": table_name, "columns": columns}

    def _is_safe_query(self, query: str) -> bool:
        """Check if query is safe (read-only)"""
        unsafe_keywords = [
            "INSERT",
            "UPDATE",
            "DELETE",
            "DROP",
            "CREATE",
            "ALTER",
            "TRUNCATE",
            "GRANT",
            "REVOKE",
        ]
        query_upper = query.upper()
        return not any(keyword in query_upper for keyword in unsafe_keywords)
