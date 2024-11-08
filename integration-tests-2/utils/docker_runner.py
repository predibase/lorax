import docker
import time
import os
import requests
from typing import Dict, Any


class DockerModelRunner:
    def __init__(self):
        self.client = docker.from_env()
        self.container = None

    def start_container(self, model_config: Dict[str, Any]) -> None:
        # Build command arguments
        cmd = ["--model-id", model_config["model_id"], "--port", "80"]

        # Add additional arguments
        if "docker_args" in model_config:
            for key, value in model_config["docker_args"].items():
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])

        data_dir = os.getenv("DATA_DIR", "/integration-test-data")
        # Start container
        self.container = self.client.containers.run(
            "ghcr.io/predibase/lorax:main",
            command=cmd,
            environment={
                "HUGGING_FACE_HUB_TOKEN": os.getenv("HUGGING_FACE_HUB_TOKEN"),
            },
            device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])],
            ports={"80": 8080},
            volumes={data_dir: {"bind": "/data", "mode": "rw"}},
            shm_size="1g",
            detach=True,
            name=f"lorax-test-{model_config['name']}-{time.time()}",
        )

    def wait_for_healthy(self, timeout: int = 300) -> None:
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get("http://localhost:8080/health")
                if response.status_code == 200:
                    return
                # check if container hasnt exited
                if self.container.status == "exited":
                    raise Exception("Container exited")
            except Exception:
                pass
            time.sleep(5)
        raise TimeoutError("Container failed to become healthy")

    def stop_container(self) -> None:
        if self.container:
            self.container.stop()
            self.container.remove()
            self.container = None
