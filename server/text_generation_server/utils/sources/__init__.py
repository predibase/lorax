from .hub import EntryNotFoundError, LocalEntryNotFoundError, RevisionNotFoundError, weight_files, download_weights, weight_hub_files, HubModelSource
from .s3 import S3ModelSource


def get_model_source(source, model_id, revision, extension):
    if source == "hub":
        return HubModelSource(model_id, revision, extension)
    elif source == "s3":
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