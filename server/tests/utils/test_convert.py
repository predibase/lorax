from pathlib import Path

import pytest
import torch

from lorax_server.utils.convert import convert_files
from lorax_server.utils.errors import NanWeightsError
from lorax_server.utils.sources.hub import (
    download_weights,
    weight_files,
    weight_hub_files,
)


def test_convert_files():
    model_id = "bigscience/bloom-560m"
    pt_filenames = weight_hub_files(model_id, extension=".bin")
    local_pt_files = download_weights(pt_filenames, model_id)
    local_st_files = [
        p.parent / f"{p.stem.lstrip('pytorch_')}.safetensors" for p in local_pt_files
    ]
    convert_files(local_pt_files, local_st_files, discard_names=[])

    found_st_files = weight_files(model_id)

    assert all([p in found_st_files for p in local_st_files])


def test_convert_files_nan_error(tmpdir):
    model_id = "bigscience/bloom-560m"
    pt_filenames = weight_hub_files(model_id, extension=".bin")
    local_pt_files = download_weights(pt_filenames, model_id)
    local_st_files = [
        p.parent / f"{p.stem.lstrip('pytorch_')}.safetensors" for p in local_pt_files
    ]

    # Introduce NaN to the first tensor in the first file
    pt_file = local_pt_files[0]
    with open(pt_file, "rb") as f:
        state_dict = torch.load(f, map_location="cpu")
        state_dict[list(state_dict.keys())[0]].fill_(float("nan"))

    # Write the corrupted state to a new temporary file
    pt_file = Path(tmpdir) / pt_file.name
    with open(pt_file, "wb") as f:
        torch.save(state_dict, f)

    # Replace the first file with the corrupted file
    local_pt_files[0] = pt_file
    
    with pytest.raises(NanWeightsError):
        convert_files(local_pt_files, local_st_files, discard_names=[])
