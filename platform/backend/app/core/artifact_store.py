import os
import hashlib
from pathlib import Path
import shutil

class ArtifactStore:
    """WP-245: Checksummed external artifact store."""
    
    BASE_DIR = Path.home() / ".learningos" / "artifacts"
    
    @classmethod
    def initialize(cls):
        cls.BASE_DIR.mkdir(parents=True, exist_ok=True)
        
    @classmethod
    def store_artifact(cls, source_path: Path) -> str:
        """Stores a file in the artifact store and returns its SHA256 checksum."""
        cls.initialize()
        
        sha256 = hashlib.sha256()
        with open(source_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256.update(byte_block)
        
        checksum = sha256.hexdigest()
        dest_path = cls.BASE_DIR / checksum
        
        if not dest_path.exists():
            shutil.copy2(source_path, dest_path)
            
        return checksum
        
    @classmethod
    def get_artifact_path(cls, checksum: str) -> Path:
        """Returns the local path to the stored artifact."""
        path = cls.BASE_DIR / checksum
        if not path.exists():
            raise FileNotFoundError(f"Artifact {checksum} not found")
        return path
