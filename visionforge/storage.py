"""Production S3 client and filesystem adapter with identical byte APIs."""
import hashlib
import os
from pathlib import Path
from urllib.parse import quote

from .config import Config


class LocalStorage:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Invalid object key: outside storage root")
        return path

    def put(self, key: str, data: bytes, *, content_type: str = "image/png") -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def iter_keys(self, prefix: str = ""):
        subdir = self._path(prefix)
        if subdir.is_file():
            yield subdir.relative_to(self.root).as_posix()
        elif subdir.exists():
            for path in subdir.rglob("*"):
                if path.is_file():
                    yield path.relative_to(self.root).as_posix()

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    def healthcheck(self) -> dict:
        return {"backend": "local", "root": str(self.root), "writable": os.access(self.root, os.W_OK)}

    def presigned_url(self, key: str, expires: int = 3600) -> str:
        return self._path(key).as_uri()


class S3Storage:
    """Supports real AWS S3, MinIO and other S3-compatible endpoints.

    Credential handling delegates to boto3's standard environment/profile/IAM
    provider chain; no credential material is embedded in code.
    """
    def __init__(self, config: Config, client=None):
        import boto3
        from botocore.config import Config as BotoConfig
        self.bucket = config.bucket
        self.client = client or boto3.client(
            "s3", endpoint_url=config.endpoint, region_name=config.region,
            config=BotoConfig(s3={"addressing_style": config.addressing_style},
                              retries={"max_attempts": 5, "mode": "standard"}),
        )

    def put(self, key: str, data: bytes, *, content_type: str = "image/png") -> None:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)

    def get(self, key: str) -> bytes:
        resp = self.client.get_object(Bucket=self.bucket, Key=key)
        with resp["Body"] as body:
            return body.read()

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def iter_keys(self, prefix: str = ""):
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                yield item["Key"]

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def healthcheck(self) -> dict:
        self.client.head_bucket(Bucket=self.bucket)
        return {"backend": "s3", "bucket": self.bucket, "reachable": True}

    def presigned_url(self, key: str, expires: int = 3600) -> str:
        return self.client.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=expires
        )


class CachedStorage:
    """Read-through disk cache for avoiding repeated GET calls across epochs.

    Cache assumes original keys are immutable. Use a new cache directory after
    replacing existing object keys. Cache content is not a credential store.
    """
    def __init__(self, remote, cache_root: Path):
        self.remote = remote
        self.cache_root = Path(cache_root)

    def get(self, key: str) -> bytes:
        digest = hashlib.sha256(key.encode()).hexdigest()
        path = self.cache_root / digest[:2] / digest
        if path.exists():
            return path.read_bytes()
        data = self.remote.get(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
        return data


def make_storage(config: Config):
    if config.storage_backend == "local":
        return LocalStorage(config.local_storage_dir)
    if config.storage_backend == "s3":
        return S3Storage(config)
    raise ValueError(f"Unknown storage backend {config.storage_backend!r}")
