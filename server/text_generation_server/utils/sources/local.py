import os
import time
from datetime import timedelta
from typing import Optional, List, Any

from loguru import logger
from pathlib import Path
import boto3
from botocore.config import Config
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE


from huggingface_hub.utils import (
    LocalEntryNotFoundError,
    EntryNotFoundError,
)

from .s3 import get_s3_model_local_dir
from .source import BaseModelSource, try_to_load_from_cache


class LocalModelSource(BaseModelSource):
    def __init__(self, model_id: str, revision: Optional[str] = "", extension: str = ".safetensors"):
        if len(model_id) < 5:
            raise ValueError(f"model_id '{model_id}' is too short for prefix filtering")

        # TODO: add support for revisions of the same model
        self.model_id = model_id
        self.revision = revision
        self.extension = extension

    def remote_weight_files(self, extension: str = None):
        return []

    def weight_files(self, extension: str = None):
        model_id = self.model_id
        extension = extension or self.extension

        local_path = get_s3_model_local_dir(model_id)
        if local_path.exists() and local_path.is_dir():
            local_files = list(local_path.glob(f"*{extension}"))
            if not local_files:
                raise FileNotFoundError(
                    f"No local weights found in {model_id} with extension {extension}"
                )
            return local_files
        
        raise FileNotFoundError(
            f"No local weights found in {model_id} with extension {extension}"
        )
    
    def download_weights(self, filenames: List[str]):
        return []

    def download_model_assets(self):
        return []

    def get_local_path(self, model_id: str):
        return get_s3_model_local_dir(model_id)