from .hub import EntryNotFoundError, LocalEntryNotFoundError, RevisionNotFoundError, weight_files, download_weights, weight_hub_files, HubModelSource
from .s3 import S3ModelSource

S3 = "s3"
HUB = "hub"


def get_model_source(source, model_id, revision, extension):
    if source == HUB:
        return HubModelSource(model_id, revision, extension)
    elif source == S3:
        return S3ModelSource(model_id, revision, extension)
    else:
        raise ValueError(f"Unknown source {source}")


__all__ = [
    "download_weights",
    "weight_hub_files",
    "weight_files",
    "get_model_source",
    "EntryNotFoundError",
    "LocalEntryNotFoundError",
    "RevisionNotFoundError",
]