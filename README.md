# SingleStore MCP Server

A Model Context Protocol (MCP) server that enables Claude to interact with SingleStore databases, providing secure read-only access with support for vector search and advanced analytics.

## Features

- 🔍 **SQL Query Execution**: Safe, read-only SQL queries with validation
- 🧮 **Vector Search**: Native SingleStore vector similarity search for AI/ML workloads
- 📊 **Schema Exploration**: Browse tables, columns, and relationships
- 🚀 **Performance Analysis**: Query execution plans and profiling
- 🔒 **Security**: SQL injection prevention, rate limiting, and authentication
- ☁️ **Cloud Deployment**: Ready for DigitalOcean App Platform

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/singlestore-mcp-server
cd singlestore-mcp-server

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"