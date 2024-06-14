#!/usr/bin/env python
''' Contains the handler function that will be called by the serverless. '''

from typing import Generator
import runpod
import os
import time

# For download the weights
from lorax import Client

# Prepare global variables
JOBS = set()
TGI_LOCAL_PORT = int(os.environ.get('TGI_LOCAL_PORT', 8080))
url = "http://127.0.0.1:{}".format(TGI_LOCAL_PORT)
# Create the client
client = Client(url)
print(url)
# Wait for the hugging face TGI worker to start running.
while True:
    try:
        client.generate("Why is the sky blue?", max_new_tokens=1).generated_text
        print("Successfully cold booted the hugging face text generation inference server!")

        # Break from the while loop
        break

    except Exception as e:
        print(e)
        print("The hugging face text generation inference server is still cold booting...")
        time.sleep(5)

def concurrency_controller() -> bool:
    # Handle at most 1024 jobs at a time.
    return len(JOBS) > 1024

async def handler_streaming(job: dict) -> Generator[dict[str, list], None, None]:
    '''
    This is the handler function that will be called by the serverless.
    ''' 
    # Get job input
    job_input = job['input']

    # TODO get stream yes/no and call the client based on that...?
    # TODO get the auth token or whatever 
    # TODO figure out how to do auth here - maybe we start it with a secret
    # and in istio-land we inject the correct secret in requests 
    # if the user is auth'ed properly for the resource? 
    # TODO handle key timeouts 
    # Add job to the set.
    JOBS.add(job['id'])

    # Streaming case
    for response in client.generate_stream(**job_input):
        if not response.token.special:
            yield response

    # Remove job from the set.
    JOBS.remove(job['id'])

# Start the serverless worker with appropriate settings
print("Starting the TGI serverless worker with streaming enabled.")
runpod.serverless.start({
    "handler": handler_streaming, 
    "concurrency_controller": concurrency_controller, 
    "return_aggregate_stream": True
})
