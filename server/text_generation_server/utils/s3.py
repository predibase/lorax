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
    breakpoint()
    cache = _get_local_cache_path()
    model_path = cache / model_id
    if not model_path.is_dir():
        return None
    cached_file = model_path / filename
    return cached_file if cached_file.is_file() else None


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


def download_weights_from_s3(
    filenames: List[str], model_id: str
) -> List[Path]:
    """Download the safetensors files from the s3"""
    def download_file(filename, tries=5, backoff: int = 5):
        local_file = try_to_load_from_cache(model_id, filename)
        if local_file is not None:
            logger.info(f"File {filename} already present in cache.")
            return Path(local_file)
        for i in range(tries):
            try:
                logger.info(f"Download file: {filename}")
                start_time = time.time()
                local_file_path = _get_local_cache_path() / filename
                # ensure cache dir exists and create it if needed
                local_file_path.parent.mkdir(parents=True, exist_ok=True)
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
            model_id, filename=filename
        )
        if cache_file is None:
            raise LocalEntryNotFoundError(
                f"File {filename} of model {model_id} not found in "
                f"{_get_local_cache_path()}. "
                f"Please run `text-generation-server download-weights {model_id}` first."
            )
        files.append(cache_file)

    return files


def main():
    # test s3 files
    filenames = weight_s3_files("opt-350m", extension=".bin")
    assert filenames == ["pytorch_model.bin"]

    # test errors
    try:
        weight_s3_files("opt-350m", extension=".errors")
    except EntryNotFoundError as e:
        print("Got correct error")

    # test downloads
    model_id = "opt-350m"
    filenames = weight_s3_files(model_id, extension=".bin")
    files = download_weights_from_s3(filenames, model_id)
    local_files = weight_files_s3("opt-350m", extension=".bin")
    assert files == local_files

if __name__ == "__main__":
    main()
