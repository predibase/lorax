import os
from typing import Optional, List
from pathlib import Path
from loguru import logger

from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE


def try_to_load_from_cache(
    model_id: str, revision: Optional[str], filename: str
) -> Optional[Path]:
    """Try to load a file from the Hugging Face cache"""
    if revision is None:
        revision = "main"

    object_id = model_id.replace("/", "--")
    repo_cache = Path(HUGGINGFACE_HUB_CACHE) / f"models--{object_id}"

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
    if revision and revision not in cached_shas:
        logger.info(f">>>>>>>>>>>> Revision {revision} not found in cache. Cache content: {cached_shas}\n\n")
        # No cache for this revision and we won't try to return a random revision
        return None

    # Check if file exists in cache
    cached_file = snapshots_dir / revision / filename
    return cached_file if cached_file.is_file() else None


class BaseModelSource:
    def remote_weight_files(self, extension: str = None):
        raise NotImplementedError

    def weight_files(self, extension: str = None):
        raise NotImplementedError
    
    def download_weights(self, filenames: List[str]):
        raise NotImplementedError
    
    def download_model_assets(self):
        """ The reason we need this function is that for s3 
        we need to download all the model files whereas for 
        hub we only need to download the weight files. And maybe 
        for other future sources  we might need something different. 
        So this function will take the necessary steps to download
        the needed files for any source """
        raise NotImplementedError
