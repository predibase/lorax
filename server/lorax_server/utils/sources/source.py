import json
import os
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from lorax_server.adapters.config import AdapterConfig


def try_to_load_from_cache(repo_cache: Path, revision: Optional[str], filename: str) -> Optional[Path]:
    """Try to load a file from the Hugging Face cache"""
    if revision is None:
        revision = "main"

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
        # No cache for this revision and we won't try to return a random revision
        return None

    # Check if file exists in cache
    cached_file = snapshots_dir / revision / filename
    return cached_file if cached_file.is_file() else None


class BaseModelSource:
    @property
    @abstractmethod
    def api_token(self) -> Optional[str]:
        pass

    @abstractmethod
    def remote_weight_files(self, extension: str = None):
        pass

    @abstractmethod
    def weight_files(self, extension: str = None) -> List[Path]:
        pass

    @abstractmethod
    def download_weights(self, filenames: List[str]):
        pass

    @abstractmethod
    def download_model_assets(self):
        """The reason we need this function is that for s3
        we need to download all the model files whereas for
        hub we only need to download the weight files. And maybe
        for other future sources  we might need something different.
        So this function will take the necessary steps to download
        the needed files for any source"""
        pass

    @abstractmethod
    def download_file(self, filename: str, ignore_errors: bool = False) -> Optional[Path]:
        pass

    def get_weight_bytes(self) -> int:
        total_size = 0
        for path in self.weight_files():
            fname = str(path)

            # safetensor format explained here: https://huggingface.co/docs/safetensors/en/index
            # parsing taken from: https://github.com/by321/safetensors_util/blob/main/safetensors_file.py
            st = os.stat(fname)
            if st.st_size < 8:
                raise RuntimeError(f"Length of safetensor file less than 8 bytes: {fname}")

            with open(fname, "rb") as f:
                # read header size
                b8 = f.read(8)
                if len(b8) != 8:
                    raise RuntimeError(f"Failed to read first 8 bytes of safetensor file: {fname}")

                headerlen = int.from_bytes(b8, "little", signed=False)
                if 8 + headerlen > st.st_size:
                    raise RuntimeError(f"Header extends past end of file: {fname}")

                hdrbuf = f.read(headerlen)
                header = json.loads(hdrbuf)
                metadata = header.get("__metadata__", {})
                total_size_bytes = metadata.get("total_size")
                if total_size_bytes is None:
                    # Fallback to determining this value from the data offsets
                    min_data_offset = None
                    max_data_offset = None
                    for v in header.values():
                        if not isinstance(v, dict):
                            continue

                        data_offsets = v.get("data_offsets")
                        if data_offsets is None:
                            continue

                        if min_data_offset is not None:
                            min_data_offset = min(min_data_offset, data_offsets[0])
                        else:
                            min_data_offset = data_offsets[0]

                        if max_data_offset is not None:
                            max_data_offset = max(max_data_offset, data_offsets[1])
                        else:
                            max_data_offset = data_offsets[1]

                    if min_data_offset is None or max_data_offset is None:
                        # Fallback to determining total bytes from file size
                        total_size_bytes = st.st_size
                    else:
                        total_size_bytes = max_data_offset - min_data_offset

                total_size += total_size_bytes

        return total_size

    def load_config(self) -> "AdapterConfig":
        from lorax_server.adapters import load_adapter_config

        config_path = self.download_file("config.json", ignore_errors=True)
        adapter_config_path = self.download_file("adapter_config.json", ignore_errors=True)
        return load_adapter_config(config_path, adapter_config_path, self.api_token)
