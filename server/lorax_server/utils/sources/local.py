import os
from pathlib import Path
from typing import List, Optional

from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE

from .source import BaseModelSource


def get_model_local_dir(model_id: str) -> Path:
    if os.path.isabs(model_id):
        return Path(model_id)

    repo_cache = Path(HUGGINGFACE_HUB_CACHE) / model_id
    return repo_cache


class LocalModelSource(BaseModelSource):
    def __init__(self, model_id: str, revision: Optional[str] = "", extension: str = ".safetensors"):
        if len(model_id) < 5:
            raise ValueError(f"model_id '{model_id}' is too short for prefix filtering")

        # TODO: add support for revisions of the same model
        self.model_id = model_id
        self.revision = revision
        self.extension = extension

    @property
    def api_token(self) -> Optional[str]:
        return None

    def remote_weight_files(self, extension: str = None):
        return []

    def weight_files(self, extension: str = None):
        model_id = self.model_id
        extension = extension or self.extension

        local_path = get_model_local_dir(model_id)
        if local_path.exists() and local_path.is_dir():
            local_files = list(local_path.glob(f"*{extension}"))
            if not local_files:
                raise FileNotFoundError(f"No local weights found in {model_id} with extension {extension}")
            return local_files

        raise FileNotFoundError(f"No local weights found in {model_id} with extension {extension}")

    def download_weights(self, filenames: List[str]):
        return []

    def download_model_assets(self):
        return []

    def get_local_path(self, model_id: str) -> Path:
        return get_model_local_dir(model_id)

    def download_file(self, filename: str, ignore_errors: bool = False) -> Optional[Path]:
        path = get_model_local_dir(self.model_id) / filename
        if not path.exists():
            if ignore_errors:
                return None
            raise FileNotFoundError(f"File {filename} of model {self.model_id} not found in {path}")
        return path
