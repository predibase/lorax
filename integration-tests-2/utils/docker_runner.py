import docker
import time
import os
import requests
from typing import Dict, Any
import contextlib


def _init_client():
    return docker.from_env()


def _start_container(client, model_config: Dict[str, Any]) -> None:
    # Build command arguments
    cmd = ["--model-id", model_config["model_id"], "--port", "80"]

    # Add additional arguments
    if "docker_args" in model_config:
        for key, value in model_config["docker_args"].items():
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])

    data_dir = os.getenv("DATA_DIR", "/integration-test-data")
    image_tag = os.getenv("TEST_IMAGE_TAG", "main")
    print(f"Using image tag: {image_tag}")
    # Start container
    container = client.containers.run(
        f"ghcr.io/predibase/lorax:{image_tag}",
        command=cmd,
        environment={
            "HUGGING_FACE_HUB_TOKEN": os.getenv("HUGGING_FACE_HUB_TOKEN"),
        },
        device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])],
        ports={"80": 8080},
        volumes={data_dir: {"bind": "/data", "mode": "rw"}},
        shm_size="1g",
        detach=True,
        name="lorax-integration-test",
    )
    return container


def _wait_for_healthy(container, timeout: int = 300) -> None:
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get("http://localhost:8080/health")
            if response.status_code == 200:
                return
            # check if container hasnt exited
            if container.status == "exited":
                raise Exception("Container exited")
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError("Container failed to become healthy")


def _stop_container(container) -> None:
    container.stop()
    container.remove()


@contextlib.contextmanager
def run_lorax_container(config):
    client = _init_client()
    container = _start_container(client, config)
    print("Started container")
    try:
        print("Waiting for container to be healthy")
        _wait_for_healthy(container)
        print("Container is healthy")
        yield
    finally:
        print("Stopping container")
        _stop_container(container)
        print("Container stopped")
