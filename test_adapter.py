import requests 

url = "http://localhost:8080/generate"
token = "pb_OSNm9rsOXoumcrQnst1TFQ"
headers = {"Authorization": f"Bearer {token}"}
params = {
        "inputs": "What is your name?",
        "parameters": { 
            "adapter_id": "emc_Qwen2-1.5b_e3_r16_lr0001_v24/1",
            "adapter_source": "pbase", 
            "max_new_tokens": 20,
            "temperature": 0.1,
            "api_token": "pb_OSNm9rsOXoumcrQnst1TFQ"
            }
}


resp = requests.post(url, headers=headers, json=params)
resp.raise_for_status()
print(resp.json())
