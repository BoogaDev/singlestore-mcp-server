"""SingleStore MCP Server Package"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .server import SingleStoreMCPServer
from .database import SingleStoreManager
from .tools import SingleStoreTools
from .security import SecurityManager

__all__ = [
    "SingleStoreMCPServer",
    "SingleStoreManager",
    "SingleStoreTools",
    "SecurityManager",
]
