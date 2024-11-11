import requests
from utils.docker_runner import run_lorax_container


def test_base_mistral():
    config = {
        "name": "mistral-7b",
        "model_id": "mistralai/Mistral-7B-Instruct-v0.1",
    }
    test_prompt = "[INST] What is the capital of France? [/INST]"
    with run_lorax_container(config):
        response = requests.post(
            "http://localhost:8080/generate",
            json={"inputs": test_prompt, "parameters": {"max_new_tokens": 10}},
        )
        response.raise_for_status()
        print("RESPONSE FROM LLM: ", response.json())
        assert len(response.json()["generated_text"]) > 0


def test_base_llama_3_1_8b():
    config = {
        "name": "llama-3-1-8b",
        "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    }
    test_prompt = "[INST] What is the capital of France? [/INST]"
    with run_lorax_container(config):
        response = requests.post(
            "http://localhost:8080/generate",
            json={"inputs": test_prompt, "parameters": {"max_new_tokens": 10}},
        )
        response.raise_for_status()
        print("RESPONSE FROM LLM: ", response.json())
        assert len(response.json()["generated_text"]) > 0


def test_base_qwen_2_1_5b():
    config = {"name": "qwen-2-1-5b", "model_id": "predibase/Qwen2-1.5B-Instruct-dequantized"}
    test_prompt = "[INST] What is the capital of France? [/INST]"
    with run_lorax_container(config):
        response = requests.post(
            "http://localhost:8080/generate",
            json={"inputs": test_prompt, "parameters": {"max_new_tokens": 10}},
        )
        response.raise_for_status()
        print("RESPONSE FROM LLM: ", response.json())
        assert len(response.json()["generated_text"]) > 0
