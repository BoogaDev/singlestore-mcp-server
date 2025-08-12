"""SingleStore MCP Server Package"""

__version__ = "1.0.0"
__author__ = "Brandon Alpert"
__email__ = "brandon@booga.io"

# Import only the modules that don't depend on MCP
from .database import SingleStoreManager
from .tools import SingleStoreTools
from .security import SecurityManager

# Try to import the server, but don't fail if MCP is missing
try:
    from .server import SingleStoreMCPServer

    __all__ = [
        "SingleStoreMCPServer",
        "SingleStoreManager",
        "SingleStoreTools",
        "SecurityManager",
    ]
except ImportError as e:
    print(f"Warning: Could not import SingleStoreMCPServer: {e}")
    __all__ = [
        "SingleStoreManager",
        "SingleStoreTools",
        "SecurityManager",
    ]
