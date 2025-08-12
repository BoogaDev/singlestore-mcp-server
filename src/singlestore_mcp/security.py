import os
import hmac
import hashlib
import time
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SecurityManager:
    def __init__(self):
        self.api_key = os.getenv("MCP_API_KEY")
        self.rate_limits = {}
        self.max_requests_per_minute = 100

    async def validate_auth(self, auth_header: Optional[str]) -> bool:
        """Validate authentication header"""
        if not self.api_key:
            # No authentication required if API key not set
            return True

        if not auth_header:
            return False

        # Extract bearer token
        if not auth_header.startswith("Bearer "):
            return False

        token = auth_header[7:]

        # Validate token (simple comparison, use JWT in production)
        return hmac.compare_digest(token, self.api_key)

    async def validate_request(self, tool_name: str, arguments: Dict[str, Any]) -> bool:
        """Validate and sanitize request"""
        # Check rate limiting
        client_id = arguments.get("client_id", "default")
        if not self._check_rate_limit(client_id):
            logger.warning(f"Rate limit exceeded for client: {client_id}")
            return False

        # Validate SQL queries
        if tool_name in ["query_database", "analyze_query"]:
            sql = arguments.get("sql", "")
            if not self._validate_sql(sql):
                logger.warning(f"Invalid SQL query: {sql[:100]}...")
                return False

        return True

    def _check_rate_limit(self, client_id: str) -> bool:
        """Simple rate limiting implementation"""
        current_time = time.time()
        minute_ago = current_time - 60

        if client_id not in self.rate_limits:
            self.rate_limits[client_id] = []

        # Clean old requests
        self.rate_limits[client_id] = [
            t for t in self.rate_limits[client_id] if t > minute_ago
        ]

        # Check limit
        if len(self.rate_limits[client_id]) >= self.max_requests_per_minute:
            return False

        # Add current request
        self.rate_limits[client_id].append(current_time)
        return True

    def _validate_sql(self, sql: str) -> bool:
        """Additional SQL validation"""
        # Check for SQL injection patterns
        dangerous_patterns = [
            "--;",
            "/*",
            "*/",
            "xp_",
            "sp_",
            "0x",
            "UNION ALL",
            "UNION SELECT",
            "OR 1=1",
            "' OR '1'='1",
        ]

        sql_upper = sql.upper()
        for pattern in dangerous_patterns:
            if pattern.upper() in sql_upper:
                return False

        # Check query length
        if len(sql) > 10000:
            return False

        return True
