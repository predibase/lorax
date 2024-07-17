import os

import pytest
from huggingface_hub.utils import RepositoryNotFoundError

from lorax_server.adapters.utils import download_adapter_weights
from lorax_server.utils.sources import HUB


def test_download_private_adapter_hf():
    # store and unset HUGGING_FACE_HUB_TOKEN from the environment
    token = os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
    assert token is not None, "HUGGING_FACE_HUB_TOKEN must be set in the environment to run this test"

    # verify download fails without the token set
    with pytest.raises(RepositoryNotFoundError):
        download_adapter_weights("predibase/test-private-lora", HUB, api_token=None)

    # pass in the token and verify download succeeds
    download_adapter_weights("predibase/test-private-lora", HUB, api_token=token)

    # set the token back in the environment
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
