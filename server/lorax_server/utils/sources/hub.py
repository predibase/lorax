import os
import time
from datetime import timedelta
from pathlib import Path
from typing import List, Optional

from huggingface_hub import HfApi, file_download, hf_hub_download
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from huggingface_hub.utils import (
    EntryNotFoundError,  # Import here to ease try/except in other part of the lib
    LocalEntryNotFoundError,
)
from loguru import logger

from .source import BaseModelSource, try_to_load_from_cache

WEIGHTS_CACHE_OVERRIDE = os.getenv("WEIGHTS_CACHE_OVERRIDE", None)
HF_HUB_OFFLINE = os.environ.get("HF_HUB_OFFLINE", "0").lower() in ["true", "1", "yes"]


def get_hub_model_local_dir(model_id: str) -> Path:
    object_id = model_id.replace("/", "--")
    repo_cache = Path(HUGGINGFACE_HUB_CACHE) / f"models--{object_id}"
    return repo_cache


def _cached_weight_files(model_id: str, revision: Optional[str], extension: str) -> List[str]:
    """Guess weight files from the cached revision snapshot directory"""
    d = _get_cached_revision_directory(model_id, revision)
    if not d:
        return []
    filenames = _weight_files_from_dir(d, extension)
    return filenames


def _get_cached_revision_directory(model_id: str, revision: Optional[str]) -> Optional[Path]:
    if revision is None:
        revision = "main"

    repo_cache = Path(HUGGINGFACE_HUB_CACHE) / Path(file_download.repo_folder_name(repo_id=model_id, repo_type="model"))

    if not repo_cache.is_dir():
        # No cache for this model
        return None

    refs_dir = repo_cache / "refs"
    snapshots_dir = repo_cache / "snapshots"

    # Resolve refs (for instance to convert main to the associated commit sha)
    if refs_dir.is_dir():
        revision_file = refs_dir / revision
        if revision_file.exists():
            with revision_file.open() as f:
                revision = f.read()

    # Check if revision folder exists
    if not snapshots_dir.exists():
        return None
    cached_shas = os.listdir(snapshots_dir)
    if revision not in cached_shas:
        # No cache for this revision and we won't try to return a random revision
        return None

    return snapshots_dir / revision


def _weight_files_from_dir(d: Path, extension: str) -> List[str]:
    # os.walk: do not iterate, just scan for depth 1, not recursively
    # see _weight_hub_files_from_model_info, that's also what is
    # done there with the len(s.rfilename.split("/")) == 1 condition
    root, _, files = next(os.walk(str(d)))
    filenames = [
        os.path.join(root, f)
        for f in files
        if f.endswith(extension) and "arguments" not in f and "args" not in f and "training" not in f
    ]
    return filenames


def weight_hub_files(
    model_id: str,
    revision: Optional[str] = None,
    extension: str = ".safetensors",
    api_token: Optional[str] = None,
    embedding_dim: Optional[int] = None,
) -> List[str]:
    """Get the weights filenames on the hub"""
    if HF_HUB_OFFLINE:
        filenames = _cached_weight_files(model_id, revision, extension)
    else:
        api = get_hub_api(token=api_token)
        info = api.model_info(model_id, revision=revision)
        if embedding_dim is not None:
            filenames = [
                s.rfilename
                for s in info.siblings
                if s.rfilename.endswith(extension)
                and len(s.rfilename.split("/")) <= 2
                and "arguments" not in s.rfilename
                and "args" not in s.rfilename
                and "training" not in s.rfilename
            ]
            # Only include the layer for the correct embedding dim
            embedding_tensor_file = f"2_Dense_{embedding_dim}/model.safetensors"
            if embedding_tensor_file not in filenames:
                raise ValueError(f"No embedding tensor file found for embedding dim {embedding_dim}")
            filenames = [
                filename for filename in filenames if len(filename.split("/")) < 2 or filename == embedding_tensor_file
            ]
        else:
            filenames = [
                s.rfilename
                for s in info.siblings
                if s.rfilename.endswith(extension)
                and len(s.rfilename.split("/")) == 1
                and "arguments" not in s.rfilename
                and "args" not in s.rfilename
                and "training" not in s.rfilename
            ]

    if not filenames:
        raise EntryNotFoundError(
            f"No {extension} weights found for model {model_id} and revision {revision}.",
            None,
        )

    return filenames


def weight_files(
    model_id: str,
    revision: Optional[str] = None,
    extension: str = ".safetensors",
    api_token: Optional[str] = None,
    embedding_dim: Optional[int] = None,
) -> List[Path]:
    """Get the local files"""
    # Local model
    if Path(model_id).exists() and Path(model_id).is_dir():
        local_files = list(Path(model_id).glob(f"*{extension}"))
        if not local_files:
            raise FileNotFoundError(f"No local weights found in {model_id} with extension {extension}")
        return local_files

    try:
        filenames = weight_hub_files(model_id, revision, extension, api_token, embedding_dim)
    except EntryNotFoundError as e:
        if extension != ".safetensors":
            raise e
        # Try to see if there are pytorch weights
        pt_filenames = weight_hub_files(model_id, revision, extension=".bin", api_token=api_token)
        # Change pytorch extension to safetensors extension
        # It is possible that we have safetensors weights locally even though they are not on the
        # hub if we converted weights locally without pushing them
        filenames = [f"{Path(f).stem.lstrip('pytorch_')}.safetensors" for f in pt_filenames]

    if WEIGHTS_CACHE_OVERRIDE is not None:
        files = []
        for filename in filenames:
            p = Path(WEIGHTS_CACHE_OVERRIDE) / filename
            if not p.exists():
                raise FileNotFoundError(f"File {p} not found in {WEIGHTS_CACHE_OVERRIDE}.")
            files.append(p)
        return files

    repo_cache = get_hub_model_local_dir(model_id)
    files = []
    for filename in filenames:
        cache_file = try_to_load_from_cache(repo_cache, revision=revision, filename=filename)
        if cache_file is None:
            raise LocalEntryNotFoundError(
                f"File {filename} of model {model_id} not found in "
                f"{os.getenv('HUGGINGFACE_HUB_CACHE', 'the local cache')}. "
                f"Please run `lorax-server download-weights {model_id}` first."
            )
        files.append(cache_file)

    return files


def download_weights(
    filenames: List[str],
    model_id: str,
    revision: Optional[str] = None,
    api_token: Optional[str] = None,
) -> List[Path]:
    """Download the safetensors files from the hub"""

    def download_file(filename, tries=5, backoff: int = 5):
        repo_cache = get_hub_model_local_dir(model_id)
        local_file = try_to_load_from_cache(repo_cache, revision, filename)
        if local_file is not None:
            logger.info(f"File {filename} already present in cache.")
            return Path(local_file)

        for i in range(tries):
            try:
                logger.info(f"Download file: {filename}")
                start_time = time.time()
                local_file = hf_hub_download(
                    filename=filename,
                    repo_id=model_id,
                    revision=revision,
                    local_files_only=False,
                    token=api_token,
                )
                logger.info(f"Downloaded {local_file} in {timedelta(seconds=int(time.time() - start_time))}.")
                return Path(local_file)
            except Exception as e:
                if i + 1 == tries:
                    raise e
                logger.error(e)
                logger.info(f"Retrying in {backoff} seconds")
                time.sleep(backoff)
                logger.info(f"Retry {i + 1}/{tries - 1}")

    # We do this instead of using tqdm because we want to parse the logs with the launcher
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


class HubModelSource(BaseModelSource):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        extension: str = ".safetensors",
        api_token: Optional[str] = None,
        embedding_dim: Optional[int] = None,
    ):
        self.model_id = model_id
        self.revision = revision
        self.extension = extension
        self._api_token = api_token
        self.embedding_dim = embedding_dim

    @property
    def api_token(self) -> Optional[str]:
        return self._api_token

    def remote_weight_files(self, extension: str = None):
        extension = extension or self.extension
        return weight_hub_files(self.model_id, self.revision, extension, self.api_token, self.embedding_dim)

    def weight_files(self, extension=None):
        extension = extension or self.extension
        return weight_files(self.model_id, self.revision, extension, self.api_token, self.embedding_dim)

    def download_weights(self, filenames):
        return download_weights(filenames, self.model_id, self.revision, self.api_token)

    def download_model_assets(self):
        filenames = self.remote_weight_files()
        return self.download_weights(filenames)

    def download_file(self, filename: str, ignore_errors: bool = False) -> Optional[Path]:
        try:
            return Path(hf_hub_download(self.model_id, revision=None, filename=filename, token=self.api_token))
        except Exception as e:
            if ignore_errors:
                return None
            raise e


def get_hub_api(token: Optional[str] = None) -> HfApi:
    if token == "" and bool(int(os.environ.get("LORAX_USE_GLOBAL_HF_TOKEN", "0"))):
        # User initialized LoRAX to fallback to global HF token if request token is empty
        token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return HfApi(token=token)
