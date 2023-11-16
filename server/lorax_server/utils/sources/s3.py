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

from .source import BaseModelSource, try_to_load_from_cache


def _get_bucket_resource():
    """Get the s3 client"""
    config = Config(
        retries=dict(
            max_attempts=5,
            mode="standard",
        )
    )
    s3 = boto3.resource('s3', config=config)
    bucket = os.getenv("PREDIBASE_MODEL_BUCKET")
    if not bucket:
        raise ValueError("PREDIBASE_MODEL_BUCKET environment variable is not set")
    return s3.Bucket(bucket)


def get_s3_model_local_dir(model_id: str):
    object_id = model_id.replace("/", "--")
    repo_cache = Path(HUGGINGFACE_HUB_CACHE) / f"models--{object_id}" / "snapshots"
    return repo_cache


def weight_s3_files(
    bucket: Any, model_id: str, extension: str = ".safetensors"
) -> List[str]:
    """Get the weights filenames from s3"""
    model_files = bucket.objects.filter(Prefix=model_id)
    filenames = [f.key.removeprefix(model_id).lstrip("/") for f in model_files if f.key.endswith(extension)]
    if not filenames:
        raise EntryNotFoundError(
            f"No {extension} weights found for model {model_id}",
            None,
        )
    return filenames


def download_files_from_s3(
    bucket: Any, filenames: List[str], model_id: str, revision: str = "",
) -> List[Path]:
    """Download the safetensors files from the s3"""
    def download_file(filename):
        repo_cache = get_s3_model_local_dir(model_id)
        local_file = try_to_load_from_cache(repo_cache, revision, filename)
        if local_file is not None:
            logger.info(f"File {filename} already present in cache.")
            return Path(local_file)
        logger.info(f"Download file: {filename}")
        start_time = time.time()
        local_file_path = get_s3_model_local_dir(model_id) / filename
        # ensure cache dir exists and create it if needed
        local_file_path.parent.mkdir(parents=True, exist_ok=True)
        model_id_path = Path(model_id)
        bucket_file_name = model_id_path / filename
        logger.info(f"Downloading file {bucket_file_name} to {local_file_path}")
        bucket.download_file(str(bucket_file_name), str(local_file_path))
        # TODO: add support for revision
        logger.info(
            f"Downloaded {local_file_path} in {timedelta(seconds=int(time.time() - start_time))}."
        ) 
        if not local_file_path.is_file():
            raise FileNotFoundError(f"File {local_file_path} not found")
        return local_file_path

    start_time = time.time()
    files = []
    for i, filename in enumerate(filenames):
        # TODO: clean up name creation logic
        if not filename:
            continue
        file = download_file(filename)

        elapsed = timedelta(seconds=int(time.time() - start_time))
        remaining = len(filenames) - (i + 1)
        eta = (elapsed / (i + 1)) * remaining if remaining > 0 else 0

        logger.info(f"Download: [{i + 1}/{len(filenames)}] -- ETA: {eta}")
        files.append(file)

    return files


def weight_files_s3(
    bucket: Any, model_id: str, revision: str = "", extension: str = ".safetensors"
) -> List[Path]:
    """Get the local files"""
    # Local model
    local_path = get_s3_model_local_dir(model_id)
    if local_path.exists() and local_path.is_dir():
        local_files = list(local_path.glob(f"*{extension}"))
        if not local_files:
            raise FileNotFoundError(
                f"No local weights found in {model_id} with extension {extension}"
            )
        return local_files

    try:
        filenames = weight_s3_files(bucket, model_id, extension)
    except EntryNotFoundError as e:
        if extension != ".safetensors":
            raise e
        # Try to see if there are pytorch weights
        pt_filenames = weight_s3_files(bucket, model_id, extension=".bin")
        # Change pytorch extension to safetensors extension
        # It is possible that we have safetensors weights locally even though they are not on the
        # hub if we converted weights locally without pushing them
        filenames = [
            f"{Path(f).stem.lstrip('pytorch_')}.safetensors" for f in pt_filenames
        ]

    repo_cache = get_s3_model_local_dir(model_id)
    files = []
    for filename in filenames:
        cache_file = try_to_load_from_cache(
            repo_cache, revision, filename
        )
        if cache_file is None:
            raise LocalEntryNotFoundError(
                f"File {filename} of model {model_id} not found in "
                f"{HUGGINGFACE_HUB_CACHE}. "
                f"Please run `lorax-server download-weights {model_id}` first."
            )
        files.append(cache_file)

    return files


def download_model_from_s3(bucket: Any, model_id: str, extension: str = ".safetensors"):
    model_files = bucket.objects.filter(Prefix=model_id)
    # ensure that only one model is retrieved by filtering on the first dir of the path
    total_models = set([Path(f.key).parts[0] for f in model_files])
    if len(total_models) > 1:
        raise ValueError(f"Multiple models found for model_id {model_id}")

    # need to filter out the empty name
    filenames = [f.key.removeprefix(model_id).lstrip("/") for f in model_files]
    logger.info(filenames)
    download_files_from_s3(bucket, filenames, model_id)
    logger.info(f"Downloaded {len(filenames)} files")
    logger.info(f"Contents of the cache folder: {os.listdir(get_s3_model_local_dir(model_id))}")

    # Raise an error if none of the files we downloaded have the correct extension
    filenames_with_extension = [f for f in model_files if f.key.endswith(extension)]
    if not filenames_with_extension:
        raise EntryNotFoundError(
            f"No {extension} weights found for model {model_id}",
            None,
        )


class S3ModelSource(BaseModelSource):
    def __init__(self, model_id: str, revision: Optional[str] = "", extension: str = ".safetensors"):
        if len(model_id) < 5:
            raise ValueError(f"model_id '{model_id}' is too short for prefix filtering")

        # TODO: add support for revisions of the same model
        self.model_id = model_id
        self.revision = revision
        self.extension = extension
        self.bucket = _get_bucket_resource()

    def remote_weight_files(self, extension: str = None):
        extension = extension or self.extension
        return weight_s3_files(self.bucket, self.model_id, extension)

    def weight_files(self, extension: str = None):
        extension = extension or self.extension
        return weight_files_s3(self.bucket, self.model_id, self.revision, extension)
    
    def download_weights(self, filenames: List[str]):
        return download_files_from_s3(self.bucket, filenames, self.model_id, self.revision)

    def download_model_assets(self):
        return download_model_from_s3(self.bucket, self.model_id, self.extension)

    def get_local_path(self, model_id: str):
        return get_s3_model_local_dir(model_id)