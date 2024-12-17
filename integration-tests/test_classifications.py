import requests
from utils.docker_runner import run_lorax_container


def test_distilbert_ner():
    config = {
        "name": "distilbert-ner",
        "model_id": "dslim/distilbert-NER",
        "docker_args": {
            "max_input_length": 512,
            "max_batch_prefill_tokens": 512,
            "max_batch_total_tokens": 512,
            "max_total_tokens": 512,
        },
    }
    with run_lorax_container(config):
        response = requests.post(
            "http://localhost:8080/classify",
            json={
                "inputs": "Johnny supports the Golden State Warriors. He lives in London."
            },
        )
        response.raise_for_status()
        print("RESPONSE FROM CLASSIFICATION:", response.json())
        assert len(response.json()) > 0


def test_bert_ner():
    config = {
        "name": "bert-ner",
        "model_id": "magdyks/bert-base-ner",
        "docker_args": {
            "max_input_length": 512,
            "max_batch_prefill_tokens": 512,
            "max_batch_total_tokens": 512,
            "max_total_tokens": 512,
            "backend": "flashinfer",
        },
    }
    with run_lorax_container(config):
        response = requests.post(
            "http://localhost:8080/classify",
            json={
                "inputs": "Johnny supports the Golden State Warriors. He lives in London."
            },
        )
        response.raise_for_status()
        print("RESPONSE FROM CLASSIFICATION:", response.json())
        assert len(response.json()) > 0
