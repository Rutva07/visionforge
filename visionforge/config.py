"""Configuration; dotenv is optional at import time."""
import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass(frozen=True)
class Config:
    storage_backend: str = "local"
    bucket: str = "visionforge-images"
    prefix: str = "visionforge"
    endpoint: str | None = None
    region: str = "us-east-1"
    addressing_style: str = "auto"
    database_url: str = "sqlite:///./visionforge-dev.sqlite3"
    local_storage_dir: Path = Path(".local/objects")
    cache_dir: Path = Path(".cache/objects")

    @classmethod
    def from_env(cls):
        return cls(
            storage_backend=os.getenv("STORAGE_BACKEND", "local"),
            bucket=os.getenv("S3_BUCKET", "visionforge-images"),
            prefix=os.getenv("S3_PREFIX", "visionforge").strip("/"),
            endpoint=os.getenv("S3_ENDPOINT_URL") or None,
            region=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            addressing_style=os.getenv("S3_ADDRESSING_STYLE", "auto"),
            database_url=os.getenv("DATABASE_URL", "sqlite:///./visionforge-dev.sqlite3"),
            local_storage_dir=Path(os.getenv("LOCAL_STORAGE_DIR", ".local/objects")),
            cache_dir=Path(os.getenv("IMAGE_CACHE_DIR", ".cache/objects")),
        )

    def key(self, *pieces: str) -> str:
        # Every object key is generated from trusted pipeline fields.
        suffix = "/".join(p.strip("/") for p in pieces)
        return f"{self.prefix}/{suffix}" if self.prefix else suffix
