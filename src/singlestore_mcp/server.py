import asyncio
import os
import json
import logging
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, Resource, TextContent
from pydantic import BaseModel, Field
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
import uvicorn

from .database import SingleStoreManager
from .tools import SingleStoreTools
from .security import SecurityManager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class QueryArguments(BaseModel):
    sql: str = Field(description="SQL query to execute")
    database_id: Optional[str] = Field(default=None, description="Database identifier")
    timeout: Optional[int] = Field(default=30, description="Query timeout in seconds")


class VectorSearchArguments(BaseModel):
    table_name: str = Field(description="Table containing vector data")
    vector_column: str = Field(description="Name of the vector column")
    query_vector: List[float] = Field(description="Query vector for similarity search")
    limit: int = Field(default=10, description="Number of results to return")
    database_id: Optional[str] = Field(default=None, description="Database identifier")


class SingleStoreMCPServer:
    def __init__(self):
        self.server = Server("singlestore-mcp-server")
        self.db_manager = SingleStoreManager()
        self.tools = SingleStoreTools(self.db_manager)
        self.security = SecurityManager()
        self._setup_handlers()

    def _setup_handlers(self):
        """Set up all MCP protocol handlers"""

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available SingleStore tools"""
            return [
                Tool(
                    name="query_database",
                    description="Execute a read-only SQL query on SingleStore",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "sql": {
                                "type": "string",
                                "description": "SQL query to execute",
                            },
                            "database_id": {
                                "type": "string",
                                "description": "Database identifier (optional)",
                            },
                            "timeout": {
                                "type": "integer",
                                "description": "Query timeout in seconds (default: 30)",
                            },
                        },
                        "required": ["sql"],
                    },
                ),
                Tool(
                    name="vector_search",
                    description="Perform vector similarity search in SingleStore",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "table_name": {
                                "type": "string",
                                "description": "Table containing vector data",
                            },
                            "vector_column": {
                                "type": "string",
                                "description": "Name of the vector column",
                            },
                            "query_vector": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "Query vector for similarity search",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Number of results (default: 10)",
                            },
                        },
                        "required": ["table_name", "vector_column", "query_vector"],
                    },
                ),
                Tool(
                    name="list_tables",
                    description="List all tables in the SingleStore database",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "database_id": {
                                "type": "string",
                                "description": "Database identifier (optional)",
                            },
                            "include_system": {
                                "type": "boolean",
                                "description": "Include system tables (default: false)",
                            },
                        },
                    },
                ),
                Tool(
                    name="describe_table",
                    description="Get detailed information about a SingleStore table",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "table_name": {
                                "type": "string",
                                "description": "Name of the table",
                            },
                            "database_id": {
                                "type": "string",
                                "description": "Database identifier (optional)",
                            },
                        },
                        "required": ["table_name"],
                    },
                ),
                Tool(
                    name="analyze_query",
                    description="Analyze query execution plan in SingleStore",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "sql": {
                                "type": "string",
                                "description": "SQL query to analyze",
                            },
                            "database_id": {
                                "type": "string",
                                "description": "Database identifier (optional)",
                            },
                        },
                        "required": ["sql"],
                    },
                ),
                Tool(
                    name="get_table_statistics",
                    description="Get statistics for a SingleStore table",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "table_name": {
                                "type": "string",
                                "description": "Name of the table",
                            },
                            "database_id": {
                                "type": "string",
                                "description": "Database identifier (optional)",
                            },
                        },
                        "required": ["table_name"],
                    },
                ),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool execution"""
            try:
                # Validate request with security manager
                if not await self.security.validate_request(name, arguments):
                    raise ValueError("Request failed security validation")

                result = None

                if name == "query_database":
                    args = QueryArguments(**arguments)
                    result = await self.tools.execute_query(
                        args.sql, args.database_id, args.timeout
                    )

                elif name == "vector_search":
                    args = VectorSearchArguments(**arguments)
                    result = await self.tools.vector_search(
                        args.table_name,
                        args.vector_column,
                        args.query_vector,
                        args.limit,
                        args.database_id,
                    )

                elif name == "list_tables":
                    database_id = arguments.get("database_id")
                    include_system = arguments.get("include_system", False)
                    result = await self.tools.list_tables(database_id, include_system)

                elif name == "describe_table":
                    table_name = arguments.get("table_name")
                    database_id = arguments.get("database_id")
                    result = await self.tools.describe_table(table_name, database_id)

                elif name == "analyze_query":
                    args = QueryArguments(**arguments)
                    result = await self.tools.analyze_query(args.sql, args.database_id)

                elif name == "get_table_statistics":
                    table_name = arguments.get("table_name")
                    database_id = arguments.get("database_id")
                    result = await self.tools.get_table_statistics(
                        table_name, database_id
                    )

                else:
                    result = {"error": f"Unknown tool: {name}"}

                return [TextContent(type="text", text=json.dumps(result, indent=2))]

            except Exception as e:
                logger.error(f"Error executing tool {name}: {str(e)}")
                return [TextContent(type="text", text=f"Error: {str(e)}")]

        @self.server.list_resources()
        async def list_resources() -> List[Resource]:
            """List available database schemas as resources"""
            resources = []
            for db_id, db_info in self.db_manager.databases.items():
                resources.append(
                    Resource(
                        uri=f"singlestore://{db_id}/schema",
                        name=f"{db_id} Schema",
                        mimeType="application/json",
                        description=f"Schema for {db_info.get('description', db_id)}",
                    )
                )
            return resources

        @self.server.read_resource()
        async def read_resource(uri: str) -> TextContent:
            """Read database schema information"""
            if uri.startswith("singlestore://"):
                parts = uri[14:].split("/")
                if len(parts) >= 2 and parts[1] == "schema":
                    db_id = parts[0]
                    schema = await self.tools.get_full_schema(db_id)
                    return TextContent(type="text", text=json.dumps(schema, indent=2))
            return TextContent(type="text", text="Resource not found")

    async def run_local(self):
        """Run the MCP server locally (for Claude Desktop)"""
        await self.db_manager.initialize()
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(read_stream, write_stream)

    async def run_remote(self):
        """Run the MCP server remotely (for DigitalOcean deployment)"""
        await self.db_manager.initialize()

        # Create FastAPI app for SSE endpoint
        app = FastAPI(title="SingleStore MCP Server")

        @app.get("/health")
        async def health_check():
            return {"status": "healthy", "service": "singlestore-mcp-server"}

        @app.post("/sse")
        async def sse_endpoint(request: Request):
            """SSE endpoint for remote MCP connections"""
            try:
                # Validate authentication
                auth_header = request.headers.get("Authorization")
                if not await self.security.validate_auth(auth_header):
                    raise HTTPException(status_code=401, detail="Unauthorized")

                # Handle MCP protocol over SSE
                async def event_generator():
                    async for message in self.server.handle_sse():
                        yield {"data": json.dumps(message)}

                return EventSourceResponse(event_generator())

            except Exception as e:
                logger.error(f"SSE endpoint error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        # Run the FastAPI server
        config = uvicorn.Config(
            app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)), log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()


def main():
    """Entry point for the MCP server"""
    import sys

    server = SingleStoreMCPServer()

    # Check if running locally or remotely
    if "--remote" in sys.argv or os.getenv("DEPLOYMENT_MODE") == "remote":
        asyncio.run(server.run_remote())
    else:
        asyncio.run(server.run_local())


if __name__ == "__main__":
    main()
