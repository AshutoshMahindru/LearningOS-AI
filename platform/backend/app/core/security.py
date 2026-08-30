import secrets
import os

class SecurityManager:
    _local_auth_token: str = None
    
    @classmethod
    def get_or_create_local_token(cls) -> str:
        """WP-234: Generates a local, ephemeral auth token on startup."""
        if not cls._local_auth_token:
            cls._local_auth_token = secrets.token_hex(32)
        return cls._local_auth_token
        
    @classmethod
    def get_provider_secret(cls, provider: str) -> str:
        """WP-236: Securely manages provider API keys in memory."""
        # For MVP, reads from environment, but hides from arbitrary dumping
        if provider.upper() == "OPENAI":
            return os.environ.get("OPENAI_API_KEY", "")
        return ""
        
security_manager = SecurityManager()
