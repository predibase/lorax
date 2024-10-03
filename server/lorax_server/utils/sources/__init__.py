import os
from functools import lru_cache
from typing import Optional

import requests

from .hub import (
    HubModelSource,
    download_weights,
    get_hub_model_local_dir,
    weight_files,
    weight_hub_files,
)
from .local import LocalModelSource, get_model_local_dir
from .s3 import S3ModelSource, _get_bucket_and_model_id, get_s3_model_local_dir

HUB = "hub"
S3 = "s3"
LOCAL = "local"
PBASE = "pbase"

LEGACY_PREDIBASE_MODEL_URL_ENDPOINT = "/v1/models/version/name/{}"
LEGACY_PREDIBASE_MODEL_VERSION_URL_ENDPOINT = "/v1/models/version/name/{}?version={}"
PREDIBASE_ADAPTER_VERSION_URL_ENDPOINT = "/v2/repos/{}/version/{}"
PREDIBASE_GATEWAY_ENDPOINT = os.getenv("PREDIBASE_GATEWAY_ENDPOINT", "https://api.app.predibase.com")


@lru_cache(maxsize=256)
def map_pbase_model_id_to_s3(model_id: str, api_token: str) -> str:
    if api_token is None:
        raise ValueError("api_token must be provided to for a model of source pbase")
    headers = {"Authorization": f"Bearer {api_token}"}
    name_components = model_id.split("/")

    url = None
    legacy_url = None

    if len(name_components) == 1:
        name = name_components[0]
        legacy_url = PREDIBASE_GATEWAY_ENDPOINT + LEGACY_PREDIBASE_MODEL_URL_ENDPOINT.format(name)
    elif len(name_components) == 2:
        name, version = name_components
        url = PREDIBASE_GATEWAY_ENDPOINT + PREDIBASE_ADAPTER_VERSION_URL_ENDPOINT.format(name, version)
        legacy_url = PREDIBASE_GATEWAY_ENDPOINT + LEGACY_PREDIBASE_MODEL_VERSION_URL_ENDPOINT.format(name, version)
    else:
        raise ValueError(f"Invalid model id {model_id}")

    def fetch_legacy_url():
        r = requests.get(legacy_url, headers=headers)
        r.raise_for_status()
        uuid, best_run_id = r.json()["uuid"], r.json()["bestRunID"]
        return f"{uuid}/{best_run_id}/artifacts/model/model_weights/"

    if url is not None:
        try:
            # Try to retrieve data using the new endpoint.
            resp = requests.get(url, headers=headers)
            resp.raise_for_status()
        except requests.RequestException:
            # Not found in new path, fall back to legacy endpoint.
            return fetch_legacy_url()

        path = resp.json().get("adapterPath", None)
        if path is None:
            raise RuntimeError(f"Adapter {model_id} is not yet available")
        return path
    else:
        # Use legacy path only since new endpoint requires both name and version number.
        return fetch_legacy_url()


# TODO(travis): refactor into registry pattern
def get_model_source(
    source: str,
    model_id: str,
    revision: Optional[str] = None,
    extension: str = ".safetensors",
    api_token: Optional[str] = None,
    embedding_dim: Optional[int] = None,
):
    if source == HUB:
        return HubModelSource(model_id, revision, extension, api_token, embedding_dim)
    elif source == S3:
        return S3ModelSource(model_id, revision, extension)
    elif source == LOCAL:
        return LocalModelSource(model_id, revision, extension)
    else:
        raise ValueError(f"Unknown source {source}")


def get_config_path(model_id: str, source: str) -> str:
    if source == HUB:
        return model_id
    elif source == S3:
        _, model_id = _get_bucket_and_model_id(model_id)
        return get_s3_model_local_dir(model_id).as_posix()
    elif source == LOCAL:
        return get_model_local_dir(model_id).as_posix()
    else:
        raise ValueError(f"Unknown source {source}")


def get_local_dir(model_id: str, source: str):
    if source == HUB:
        return get_hub_model_local_dir(model_id)
    elif source == S3:
        return get_s3_model_local_dir(model_id)
    elif source == LOCAL:
        return get_model_local_dir(model_id)
    else:
        raise ValueError(f"Unknown source {source}")


__all__ = [
    "download_weights",
    "weight_hub_files",
    "weight_files",
    "get_model_source",
    "get_hub_model_local_dir",
    "get_s3_model_local_dir",
    "map_pbase_model_id_to_s3",
]
