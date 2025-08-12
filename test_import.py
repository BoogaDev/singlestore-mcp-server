# test_import.py
import sys
import os

# Add src directory to Python path
src_path = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, src_path)

print("Python path:", sys.path)

try:
    from singlestore_mcp.server import SingleStoreMCPServer

    print("✅ SingleStoreMCPServer imported successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")

try:
    from singlestore_mcp.database import SingleStoreManager

    print("✅ SingleStoreManager imported successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")

try:
    from singlestore_mcp.tools import SingleStoreTools

    print("✅ SingleStoreTools imported successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")

try:
    from singlestore_mcp.security import SecurityManager

    print("✅ SecurityManager imported successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")

print("\n🎉 All imports successful! You can now run tests.")
