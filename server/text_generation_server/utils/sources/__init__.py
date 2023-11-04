from pathlib import Path
from typing import Optional

from .hub import EntryNotFoundError, LocalEntryNotFoundError, RevisionNotFoundError, get_hub_model_local_dir, weight_files, download_weights, weight_hub_files, HubModelSource
from .local import LocalModelSource
from .s3 import S3ModelSource, get_s3_model_local_dir

HUB = "hub"
S3 = "s3"
LOCAL = "local"


# TODO(travis): refactor into registry pattern
def get_model_source(source: str, model_id: str, revision: Optional[str] = None, extension: str = ".safetensors"):
    if source == HUB:
        return HubModelSource(model_id, revision, extension)
    elif source == S3:
        return S3ModelSource(model_id, revision, extension)
    elif source == LOCAL:
        return LocalModelSource(model_id, revision, extension)
    else:
        raise ValueError(f"Unknown source {source}")


def get_config_path(model_id: str, source: str) -> Path:
    if source == HUB:
        return model_id
    elif source == S3:
        return get_s3_model_local_dir(model_id)
    elif source == LOCAL:
        return get_s3_model_local_dir(model_id)
    else:
        raise ValueError(f"Unknown source {source}")


def get_local_dir(model_id: str, source: str):
    if source == HUB:
        return get_hub_model_local_dir(model_id)
    elif source == S3:
        return get_s3_model_local_dir(model_id)
    elif source == LOCAL:
        return get_s3_model_local_dir(model_id)
    else:
        raise ValueError(f"Unknown source {source}")


__all__ = [
    "download_weights",
    "weight_hub_files",
    "weight_files",
    "get_model_source",
    "EntryNotFoundError",
    "LocalEntryNotFoundError",
    "RevisionNotFoundError",
    "get_hub_model_local_dir",
    "get_s3_model_local_dir",
]