from lorax import Client

endpoint_url = "http://127.0.0.1:8080"
client = Client(endpoint_url, timeout=30)

prompt = "Hello, I'm a language model, "

response = client.generate(prompt, max_new_tokens=10)

print(response.generated_text)