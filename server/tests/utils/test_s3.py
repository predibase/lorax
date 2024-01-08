import contextlib
import os
from typing import Optional

import pytest

from lorax_server.utils.sources.s3 import _get_bucket_and_model_id


@contextlib.contextmanager
def with_env_var(key: str, value: Optional[str]):
    if value is None:
        yield
        return
    
    prev = os.environ.get(key)
    try:
        os.environ[key] = value
        yield
    finally:
        if prev is None:
            del os.environ[key]
        else:
            os.environ[key] = prev


@pytest.mark.parametrize(
    "s3_path, env_var, expected_bucket, expected_model_id",
    [
        ("s3://loras/foobar", None, "loras", "foobar"),
        ("s3://loras/foo/bar", None, "loras", "foo/bar"),
        ("s3://loras/foo/bar", "bucket", "loras", "foo/bar"),
        ("loras/foobar", None, "loras", "foobar"),
        ("loras/foo/bar", None, "loras", "foo/bar"),
        ("loras/foo/bar", "bucket", "bucket", "loras/foo/bar"),
    ]
)
def test_get_bucket_and_model_id(
    s3_path: str,
    env_var: Optional[str],
    expected_bucket: str,
    expected_model_id: str,
):
    with with_env_var("PREDIBASE_MODEL_BUCKET", env_var):
        bucket, model_id = _get_bucket_and_model_id(s3_path)
    assert bucket == expected_bucket
    assert model_id == expected_model_id
