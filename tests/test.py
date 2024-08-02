from lorax import Client
import sys

pod_id = sys.argv[1]

client = Client(f"https://{pod_id}-8080.proxy.runpod.net")

response = client.generate("hello!", max_new_tokens=10)
print(response)