from typing import Optional, List
import time
from loguru import logger
import os
from datetime import timedelta
from pathlib import Path
import boto3
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE


# XXX:
# TODO: move this to env variable
MODEL_BUCKET = "magdy-test"

from huggingface_hub.utils import (
    LocalEntryNotFoundError,
    EntryNotFoundError,
)


def _get_bucket_resource():
    """Get the s3 client"""
    s3 = boto3.resource('s3')
    return s3.Bucket(MODEL_BUCKET)


def _get_local_cache_path():
    """Get the local cache path"""
    # TODO: Make the cache path consistent with the hub download cache path
    default_home = os.path.join(os.path.expanduser("~"), ".cache")
    kubellm_cache_home = os.path.expanduser(
        os.path.join(default_home, "kubellm"),
    )

    default_cache_path = os.path.join(kubellm_cache_home, "s3")
    return Path(default_cache_path)


def try_to_load_from_cache(model_id: str, filename):
    cache = _get_local_cache_path()
    model_path = cache / model_id
    if not model_path.is_dir():
        return None
    cached_file = model_path / filename
    return cached_file if cached_file.is_file() else None


def weight_s3_files(
    model_id: str, revision: Optional[str] = None, extension: str = ".safetensors"
) -> List[str]:
    """Get the weights filenames from s3"""
    bucket = _get_bucket_resource()   
    model_files = bucket.objects.filter(Prefix=model_id)
    filenames = [f.key for f in model_files if f.key.endswith(extension)]
    if not filenames:
        raise EntryNotFoundError(
            f"No {extension} weights found for model {model_id} and revision {revision}.",
            None,
        )
    return filenames


def download_weights(
    filenames: List[str], model_id: str, revision: Optional[str] = None
) -> List[Path]:
    """Download the safetensors files from the s3"""
    bucket = _get_bucket_resource()
    def download_file(filename, tries=5, backoff: int = 5):
        local_file = try_to_load_from_cache(model_id, revision, filename)
        if local_file is not None:
            logger.info(f"File {filename} already present in cache.")
            return Path(local_file)
        # TODO: add cache support
        for i in range(tries):
            try:
                logger.info(f"Download file: {filename}")
                start_time = time.time()
                local_file_path = _get_local_cache_path() / filename
                # TODO: add support for revision
                local_file = bucket.download_file(filename, str(local_file_path))
                logger.info(
                    f"Downloaded {local_file} in {timedelta(seconds=int(time.time() - start_time))}."
                )
                return Path(local_file)
            except Exception as e:
                if i + 1 == tries:
                    raise e
                time.sleep(backoff)

    start_time = time.time()
    files = []
    for i, filename in enumerate(filenames):
        file = download_file(filename)

        elapsed = timedelta(seconds=int(time.time() - start_time))
        remaining = len(filenames) - (i + 1)
        eta = (elapsed / (i + 1)) * remaining if remaining > 0 else 0

        logger.info(f"Download: [{i + 1}/{len(filenames)}] -- ETA: {eta}")
        files.append(file)

    return files


def weight_files(
    model_id: str, revision: Optional[str] = None, extension: str = ".safetensors"
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
        filenames = weight_s3_files(model_id, revision, extension)
    except EntryNotFoundError as e:
        if extension != ".safetensors":
            raise e
        # Try to see if there are pytorch weights
        pt_filenames = weight_s3_files(model_id, revision, extension=".bin")
        # Change pytorch extension to safetensors extension
        # It is possible that we have safetensors weights locally even though they are not on the
        # hub if we converted weights locally without pushing them
        filenames = [
            f"{Path(f).stem.lstrip('pytorch_')}.safetensors" for f in pt_filenames
        ]

    # TODO: add support for WEIGHTS_CACHE_OVERRIDE
    return [Path(f) for f in filenames]