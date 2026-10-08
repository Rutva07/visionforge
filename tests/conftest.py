from pathlib import Path
import pytest
import torch
from visionforge.config import Config
from visionforge.database import connect
from visionforge.storage import LocalStorage


@pytest.fixture(autouse=True)
def fast_cpu():
    torch.set_num_threads(2)


@pytest.fixture
def rig(tmp_path):
    cfg = Config(storage_backend="local", database_url=f"sqlite:///{tmp_path / 'test.sqlite'}",
                 local_storage_dir=tmp_path / "objects", cache_dir=tmp_path / "cache")
    maker = connect(cfg.database_url)
    with maker() as session:
        yield cfg, LocalStorage(cfg.local_storage_dir), session
