import os
from typing import Optional

from lorax_server.utils.sources import HUB, PBASE, S3, get_model_source, map_pbase_model_id_to_s3
from lorax_server.utils.sources.hub import get_hub_api
from lorax_server.utils.weights import download_weights


def download_adapter_weights(
    adapter_id: str,
    adapter_source: str,
    api_token: Optional[str] = None,
) -> int:
    if adapter_source == PBASE:
        api_token = api_token or os.environ.get("PREDIBASE_API_TOKEN")
        adapter_id = map_pbase_model_id_to_s3(adapter_id, api_token)
        adapter_source = S3

    if adapter_source == HUB:
        # Quick auth check on the repo against the token
        get_hub_api(token=api_token).model_info(adapter_id, revision=None)

    # fail fast if ID is not an adapter (i.e. it is a full model)
    source = get_model_source(adapter_source, adapter_id, extension=".safetensors", api_token=api_token)
    source.load_config()

    download_weights(adapter_id, source=adapter_source, api_token=api_token)

    # Calculate size of adapter to be loaded
    return source.get_weight_bytes()
