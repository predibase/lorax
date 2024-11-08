import requests
from utils.docker_runner import DockerModelRunner


def test_base_mistral():
    config = {
        "name": "mistral-7b",
        "model_id": "mistralai/Mistral-7B-Instruct-v0.1",
    }
    runner = DockerModelRunner()
    runner.start_container(config)
    runner.wait_for_healthy()
    test_prompt = "[INST] What is the capital of France? [/INST]"
    response = requests.post(
        "http://localhost:8080/generate",
        json={"inputs": test_prompt, "parameters": {"max_new_tokens": 10}},
    )
    response.raise_for_status()
    print("RESPONSE FROM LLM: ", response.json())
    assert len(response.json()["generated_text"]) > 0
    runner.stop_container()
