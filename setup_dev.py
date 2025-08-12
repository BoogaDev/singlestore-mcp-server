#!/usr/bin/env python
"""Development setup script for SingleStore MCP Server"""

import subprocess
import sys
import os


def run_command(cmd):
    """Run a command and return success status"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    print(result.stdout)
    return True


def main():
    print("Setting up SingleStore MCP Server for development...")

    # Install required packages
    packages = [
        "singlestoredb",
        "aiomysql",
        "pymysql",
        "fastapi",
        "uvicorn[standard]",
        "sse-starlette",
        "httpx",
        "pydantic",
        "pydantic-settings",
        "python-dotenv",
        "sqlparse",
        "orjson",
        "structlog",
        "python-json-logger",
        "asyncio-throttle",
        "pytest",
        "pytest-asyncio",
        "pytest-cov",
    ]

    print("\nInstalling dependencies...")
    for package in packages:
        if not run_command(f"{sys.executable} -m pip install {package}"):
            print(f"Failed to install {package}")

    # Test imports
    print("\nTesting imports...")
    sys.path.insert(0, "src")

    try:
        from singlestore_mcp.database import SingleStoreManager

        print("✅ Database module imported successfully")
    except ImportError as e:
        print(f"❌ Database import failed: {e}")

    try:
        from singlestore_mcp.tools import SingleStoreTools

        print("✅ Tools module imported successfully")
    except ImportError as e:
        print(f"❌ Tools import failed: {e}")

    try:
        from singlestore_mcp.security import SecurityManager

        print("✅ Security module imported successfully")
    except ImportError as e:
        print(f"❌ Security import failed: {e}")

    try:
        from singlestore_mcp.server import SingleStoreMCPServer

        print("✅ Server module imported successfully (with mocked MCP)")
    except ImportError as e:
        print(f"❌ Server import failed: {e}")

    print("\nSetup complete! You can now run:")
    print("  PYTHONPATH=src pytest tests/test_server.py -v")
    print("  PYTHONPATH=src python -m singlestore_mcp.server")


if __name__ == "__main__":
    main()
