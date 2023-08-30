from typing import Optional, List
import time
from loguru import logger
import os
from datetime import timedelta
from pathlib import Path
import boto3
import os
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE


from huggingface_hub.utils import (
    LocalEntryNotFoundError,
    EntryNotFoundError,
)

from .source import BaseModelSource, try_to_load_from_cache


def _get_bucket_resource():
    """Get the s3 client"""
    s3 = boto3.resource('s3')
    bucket = os.getenv("PREDIBASE_MODEL_BUCKET")
    if not bucket:
        raise ValueError("PREDIBASE_MODEL_BUCKET environment variable is not set")
    return s3.Bucket(bucket)


def get_s3_model_local_path(model_id: str):
    object_id = model_id.replace("/", "--")
    repo_cache = Path(HUGGINGFACE_HUB_CACHE) / f"models--{object_id}" / "snapshots"
    return repo_cache


def weight_s3_files(
    model_id: str, extension: str = ".safetensors"
) -> List[str]:
    """Get the weights filenames from s3"""
    bucket = _get_bucket_resource()
    model_files = bucket.objects.filter(Prefix=model_id)
    filenames = [f.key.removeprefix(model_id).lstrip("/") for f in model_files if f.key.endswith(extension)]
    if not filenames:
        raise EntryNotFoundError(
            f"No {extension} weights found for model {model_id}",
            None,
        )
    return filenames


def download_files_from_s3(
    filenames: List[str], model_id: str
) -> List[Path]:
    """Download the safetensors files from the s3"""
    def download_file(filename, tries=5, backoff: int = 5):
        local_file = try_to_load_from_cache(model_id, None, filename)
        if local_file is not None:
            logger.info(f"File {filename} already present in cache.")
            return Path(local_file)
        for i in range(tries):
            try:
                logger.info(f"Download file: {filename}")
                start_time = time.time()
                local_file_path = get_s3_model_local_path(model_id) / filename
                # ensure cache dir exists and create it if needed
                local_file_path.parent.mkdir(parents=True, exist_ok=True)
                bucket = _get_bucket_resource()
                bucket_file_name = f"{model_id}/{filename}"
                bucket.download_file(bucket_file_name, str(local_file_path))
                # TODO: add support for revision
                logger.info(
                    f"Downloaded {local_file_path} in {timedelta(seconds=int(time.time() - start_time))}."
                ) 
                if not local_file_path.is_file():
                    raise FileNotFoundError(f"File {local_file_path} not found")
                return local_file_path
            except Exception as e:
                if i + 1 == tries:
                    raise e
                time.sleep(backoff)

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
    model_id: str, extension: str = ".safetensors"
) -> List[Path]:
    """Get the local files"""
    # Local model
    if Path(model_id).exists() and Path(model_id).is_dir():
        local_files = list(Path(model_id).glob(f"*{extension}"))
        if not local_files:
            raise FileNotFoundError(
                f"No local weights found in {model_id} with extension {extension}"
            )
        return local_files

    try:
        filenames = weight_s3_files(model_id, extension)
    except EntryNotFoundError as e:
        if extension != ".safetensors":
            raise e
        # Try to see if there are pytorch weights
        pt_filenames = weight_s3_files(model_id, extension=".bin")
        # Change pytorch extension to safetensors extension
        # It is possible that we have safetensors weights locally even though they are not on the
        # hub if we converted weights locally without pushing them
        filenames = [
            f"{Path(f).stem.lstrip('pytorch_')}.safetensors" for f in pt_filenames
        ]

    files = []
    for filename in filenames:
        cache_file = try_to_load_from_cache(
            model_id, None, filename
        )
        if cache_file is None:
            raise LocalEntryNotFoundError(
                f"File {filename} of model {model_id} not found in "
                f"{HUGGINGFACE_HUB_CACHE}. "
                f"Please run `text-generation-server download-weights {model_id}` first."
            )
        files.append(cache_file)

    return files


def download_model_from_s3(model_id: str, extension: str = ".safetensors"):
    bucket = _get_bucket_resource()
    model_files = bucket.objects.filter(Prefix=model_id)
    filenames_with_extension = [f for f in model_files if f.key.endswith(extension)]
    if not filenames_with_extension:
        raise EntryNotFoundError(
            f"No {extension} weights found for model {model_id}",
            None,
        )
    filenames = [f.key.removeprefix(model_id).lstrip("/") for f in model_files]
    # need to filter out the empty name
    filenames = [f for f in filenames if len(f)]
    logger.info(filenames)
    download_files_from_s3(filenames, model_id)
    logger.info(f"Downloaded {len(filenames)} files")
    logger.info(f"Contents of the cache folder: {os.listdir('/data/models--llama2-7b-chat-hf/snapshots/')}")



class S3ModelSource(BaseModelSource):
    def __init__(self, model_id: str, revision: Optional[str] = None, extension: str = ".safetensors"):
        # TODO: add support for revisions of the same model
        self.model_id = model_id
        self.revision = revision
        self.extension = extension
    
    def remote_weight_files(self, extension: str = None):
        extension = extension or self.extension
        return weight_s3_files(self.model_id, self.extension)

    def weight_files(self, extension: str = None):
        extension = extension or self.extension
        return weight_files_s3(self.model_id, extension)
    
    def download_weights(self, filenames: List[str]):
        return download_files_from_s3(filenames, self.model_id)

    def download_model_assets(self):
        return download_model_from_s3(self.model_id, self.extension)
