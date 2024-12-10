import requests
from utils.docker_runner import run_lorax_container


def test_stella_1_5b():
    config = {
        "name": "stella-1.5b",
        "model_id": "dunzhang/stella_en_1.5B_v5",
        "docker_args": {"embedding_dim": 256},
    }
    with run_lorax_container(config):
        response = requests.post("http://localhost:8080/embed", json={"inputs": "Hello, world!"})
        response.raise_for_status()
        print("RESPONSE FROM EMBEDDING: ", response.json())
        assert len(response.json()["embeddings"]) > 0


def test_uae_large_v1_1_5b():
    config = {
        "name": "UAE-Large-V1-1.5b",
        "model_id": "WhereIsAI/UAE-Large-V1",
        "docker_args": {
            "max_input_length": 512,
            "max_batch_prefill_tokens": 512,
            "max_batch_total_tokens": 512,
            "max_total_tokens": 512,
        },
    }
    with run_lorax_container(config):
        response = requests.post("http://localhost:8080/embed", json={"inputs": "Hello, world!"})
        response.raise_for_status()
        print("RESPONSE FROM EMBEDDING: ", response.json())
        assert len(response.json()["embeddings"]) > 0
