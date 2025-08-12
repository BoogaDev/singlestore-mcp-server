import logging
from typing import Any, Dict, List, Optional
from .database import SingleStoreManager

logger = logging.getLogger(__name__)


class SingleStoreTools:
    def __init__(self, db_manager: SingleStoreManager):
        self.db_manager = db_manager

    async def execute_query(
        self, sql: str, database_id: Optional[str] = None, timeout: int = 30
    ) -> Dict[str, Any]:
        """Execute a SQL query on SingleStore"""
        try:
            async with self.db_manager.get_connection(database_id) as conn:
                results = await conn.execute(sql, timeout)

                # Limit results for large datasets
                if len(results) > 1000:
                    logger.warning(
                        f"Query returned {len(results)} rows, limiting to 1000"
                    )
                    results = results[:1000]

                return {
                    "success": True,
                    "row_count": len(results),
                    "data": results,
                    "truncated": len(results) == 1000,
                }
        except Exception as e:
            logger.error(f"Query execution error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def vector_search(
        self,
        table_name: str,
        vector_column: str,
        query_vector: List[float],
        limit: int = 10,
        database_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Perform vector similarity search in SingleStore"""
        try:
            async with self.db_manager.get_connection(database_id) as conn:
                results = await conn.vector_search(
                    table_name, vector_column, query_vector, limit
                )

                return {"success": True, "results": results, "count": len(results)}
        except Exception as e:
            logger.error(f"Vector search error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def list_tables(
        self, database_id: Optional[str] = None, include_system: bool = False
    ) -> Dict[str, Any]:
        """List all tables in the SingleStore database"""
        try:
            async with self.db_manager.get_connection(database_id) as conn:
                tables = await conn.get_tables()

                # Filter system tables if requested
                if not include_system:
                    tables = [t for t in tables if not t.startswith("_")]

                return {"success": True, "tables": tables, "count": len(tables)}
        except Exception as e:
            logger.error(f"List tables error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def describe_table(
        self, table_name: str, database_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get detailed information about a SingleStore table"""
        try:
            async with self.db_manager.get_connection(database_id) as conn:
                description = await conn.describe_table(table_name)
                return {"success": True, **description}
        except Exception as e:
            logger.error(f"Describe table error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def analyze_query(
        self, sql: str, database_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze query execution plan in SingleStore"""
        try:
            async with self.db_manager.get_connection(database_id) as conn:
                # Get EXPLAIN output
                explain_query = f"EXPLAIN {sql}"
                explain_result = await conn.execute(explain_query)

                # Try to get query profile if available
                profile = None
                if hasattr(conn, "get_query_profile"):
                    try:
                        profile = await conn.get_query_profile(sql)
                    except:
                        pass  # Profile might not be available

                return {"success": True, "explain": explain_result, "profile": profile}
        except Exception as e:
            logger.error(f"Analyze query error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def get_table_statistics(
        self, table_name: str, database_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get statistics for a SingleStore table"""
        try:
            async with self.db_manager.get_connection(database_id) as conn:
                # Get row count
                count_query = f"SELECT COUNT(*) as row_count FROM {table_name}"
                count_result = await conn.execute(count_query)

                # Get table size
                size_query = f"""
                SELECT 
                    data_length + index_length AS total_size_bytes,
                    ROUND((data_length + index_length) / 1024 / 1024, 2) AS total_size_mb
                FROM information_schema.tables
                WHERE table_name = '{table_name}'
                """
                size_result = await conn.execute(size_query)

                # Get column statistics
                stats_query = f"""
                SELECT 
                    column_name,
                    data_type,
                    is_nullable,
                    column_default
                FROM information_schema.columns
                WHERE table_name = '{table_name}'
                """
                column_stats = await conn.execute(stats_query)

                return {
                    "success": True,
                    "table_name": table_name,
                    "row_count": count_result[0]["row_count"] if count_result else 0,
                    "size": size_result[0] if size_result else {},
                    "columns": column_stats,
                }
        except Exception as e:
            logger.error(f"Get table statistics error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def get_full_schema(
        self, database_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get complete database schema"""
        try:
            async with self.db_manager.get_connection(database_id) as conn:
                tables = await conn.get_tables()
                schema = {"database_id": database_id or "default", "tables": {}}

                for table in tables[:50]:  # Limit to 50 tables
                    try:
                        schema["tables"][table] = await conn.describe_table(table)
                    except Exception as e:
                        logger.warning(f"Could not describe table {table}: {str(e)}")
                        schema["tables"][table] = {"error": str(e)}

                return schema
        except Exception as e:
            logger.error(f"Get schema error: {str(e)}")
            return {"error": str(e)}
