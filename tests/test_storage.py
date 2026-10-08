import io
import pytest
from visionforge.storage import LocalStorage, S3Storage, CachedStorage
from visionforge.config import Config


def test_local_store_and_cache(rig):
    cfg, store, _ = rig
    store.put("images/img.png", b"test bytes")
    assert store.get("images/img.png") == b"test bytes"
    assert store.exists("images/img.png")
    assert list(store.iter_keys("images")) == ["images/img.png"]
    cache = CachedStorage(store, cfg.cache_dir)
    assert cache.get("images/img.png") == b"test bytes"
    store.delete("images/img.png")
    assert cache.get("images/img.png") == b"test bytes"
    with pytest.raises(ValueError):
        store.get("../evil.txt")


class FakeClient:
    def __init__(self):
        self.items = {}
    def put_object(self, **kw):
        self.items[kw["Key"]] = kw["Body"]
    def get_object(self, **kw):
        return {"Body": io.BytesIO(self.items[kw["Key"]])}
    def head_object(self, **kw):
        if kw["Key"] not in self.items:
            from botocore.exceptions import ClientError
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
    def head_bucket(self, **kw):
        return {}
    def delete_object(self, **kw):
        self.items.pop(kw["Key"], None)
    def generate_presigned_url(self, op, Params, ExpiresIn):
        return "https://test.invalid/" + Params["Key"]
    def get_paginator(self, op):
        client = self
        class Pager:
            def paginate(self, **kw):
                yield {"Contents": [{"Key": x} for x in client.items if x.startswith(kw["Prefix"])]}
        return Pager()


def test_s3_connected_adapter_without_network():
    storage = S3Storage(Config(storage_backend="s3", bucket="test-bucket"), client=FakeClient())
    storage.put("visionforge/raw/a.png", b"png")
    assert storage.healthcheck()["reachable"]
    assert storage.get("visionforge/raw/a.png") == b"png"
    assert storage.exists("visionforge/raw/a.png")
    assert not storage.exists("no-key")
    assert list(storage.iter_keys("visionforge/raw")) == ["visionforge/raw/a.png"]
    assert storage.presigned_url("visionforge/raw/a.png").startswith("https://")
    storage.delete("visionforge/raw/a.png")
    assert not storage.exists("visionforge/raw/a.png")
