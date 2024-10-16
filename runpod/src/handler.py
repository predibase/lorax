#!/usr/bin/env python
''' Contains the handler function that will be called by the serverless. '''

from typing import Generator
import runpod
import os
import time

# For download the weights
from lorax import Client

import openai

# Prepare global variables
JOBS = set()
TGI_LOCAL_PORT = int(os.environ.get('TGI_LOCAL_PORT', 8080))
url = "http://127.0.0.1:{}".format(TGI_LOCAL_PORT)
# Create the client
client = Client(url)
api_key = os.environ.get("PREDIBASE_API_KEY", "fake")


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
    # TODO do different things based on the openai_route. Right now, just assume we are calling the openai 
    # chat completions.generate method!
    print(job_input)
    print("first print :P")
    use_openai = 'openai_route' in job_input

    # Create a new client and pass the token for every handler call
    openai_client = openai.OpenAI(
        base_url=f"{url}/v1",
        api_key=api_key
    )
    JOBS.add(job['id'])

    print(use_openai)
    if use_openai:
        # if job_input['stream'] == False:
        print(job_input)
        result = openai_client.chat.completions.create(**job_input["openai_input"]).model_dump()
        yield result
    else:
        inputs = str(job_input.get('inputs'))
        if job_input.get('_stream', False):
            del job_input['_stream']
            # Streaming case
            for response in client.generate_stream(inputs, **job_input.get('parameters', {})):
                if not response.token.special:
                    # Dump the repsonse into a dictionary
                    yield response.model_dump()
        else:
            if '_stream' in job_input:
                del job_input['_stream']
            response = client.generate(inputs, **job_input.get('parameters', {}))
            yield response.model_dump()
    # When we are called with a streaming endpoint, then we should have the field 
    # _stream = True

    # TODO handle the two openAI compatable endpoints as well...!
    # TODO get stream yes/no and call the client based on that...?
    # TODO get the auth token or whatever 
    # TODO figure out how to do auth here - maybe we start it with a secret
    # and in istio-land we inject the correct secret in requests 
    # if the user is auth'ed properly for the resource? 
    # TODO handle key timeouts 
    # Add job to the set.
    
    # Remove job from the set.
    JOBS.remove(job['id'])

# Start the serverless worker with appropriate settings
print("Starting the TGI serverless worker with streaming enabled.")
runpod.serverless.start({
    "handler": handler_streaming, 
    "concurrency_controller": concurrency_controller, 
    "return_aggregate_stream": True
})
