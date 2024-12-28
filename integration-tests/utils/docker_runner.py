import docker
import time
import os
import requests
from typing import Dict, Any
import contextlib
import logging


logger = logging.getLogger(__name__)


def _init_client():
    try:
        client = docker.DockerClient(
            base_url="unix:///var/run/docker.sock", version="auto"
        )
        # Test the connection
        client.ping()
        logger.info("Successfully connected to Docker daemon")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Docker daemon: {str(e)}")
        logger.error(f"Docker socket exists: {os.path.exists('/var/run/docker.sock')}")
        logger.error(
            f"Docker socket permissions: {oct(os.stat('/var/run/docker.sock').st_mode)}"
        )
        raise


def _start_container(client, model_config: Dict[str, Any]) -> None:
    # Build command arguments
    cmd = ["--model-id", model_config["model_id"], "--port", "80"]

    # Add additional arguments
    if "docker_args" in model_config:
        for key, value in model_config["docker_args"].items():
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])

    data_dir = os.getenv("DATA_DIR", "/integration-test-data")
    image_tag = os.getenv("TEST_IMAGE_TAG", "ghcr.io/predibase/lorax:main")
    logger.info(f"Using image tag: {image_tag}")
    # Start container
    container = client.containers.run(
        image=image_tag,
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
    logger.info("Started container")
    try:
        logger.info("Waiting for container to be healthy")
        _wait_for_healthy(container)
        startup_response = requests.get("http://localhost:8080/startup")
        startup_response.raise_for_status()
        logger.info("Container is healthy")
        yield
    finally:
        logger.info("Stopping container")
        _stop_container(container)
        logger.info("Container stopped")
