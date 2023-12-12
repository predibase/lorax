""" Example handler file. """

import runpod
from lorax import Client
# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.
from subprocess import Popen


# Launch LoRAX server in the background
p = Popen(['lorax-launcher', '--json-output', '--model-id', 'mistralai/Mistral-7B-Instruct-v0.1', '--port', '8080']) 

def handler(job):
    """ Handler function that will be used to process jobs. """
    job_input = job['input']

    prompt = job_input["prompt"]
    adapter = job_input["adapter_id"]
    endpoint_url = "http://127.0.0.1:8080"
    client = Client(endpoint_url)
    text = client.generate(prompt, adapter_id=adapter).generated_text

    return text


runpod.serverless.start({"handler": handler})
