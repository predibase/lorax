import requests

URL = "http://127.0.0.1:8081/generate"

data = {
    "inputs": "[INST] Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? [/INST]",
    "parameters": {
            "max_new_tokens": 64,
            "adapter_id": "Llama-2-7b-hf-code_alpaca_20k/7",
            "adapter_source": "pbase",
            "predibase_api_token": "asfsd",
        }
}

resp = requests.post(URL, json=data)
resp.raise_for_status()
print(resp.json())
