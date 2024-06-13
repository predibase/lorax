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

# Create the client
client = Client("http://127.0.0.1:{}".format(TGI_LOCAL_PORT))

# Wait for the hugging face TGI worker to start running.
while True:
    try:
        client.generate("Why is the sky blue?").generated_text
        print("Successfully cold booted the hugging face text generation inference server!")

        # Break from the while loop
        break

    except Exception as e:
        print("The hugging face text generation inference server is still cold booting...")
        time.sleep(5)

def concurrency_controller() -> bool:
    # Handle at most 100 jobs at a time.
    return len(JOBS) > 20

async def handler_streaming(job: dict) -> Generator[dict[str, list], None, None]:
    '''
    This is the handler function that will be called by the serverless.
    ''' 
    # Get job input
    job_input = job['input']

    # Prompts
    prompt = job_input['prompt']

    # Validate the inputs
    sampling_params = job_input.get('sampling_params', {})

    # Add job to the set.
    JOBS.add(job['id'])

     # Include metrics in the highest level for the job output for aggregrate.
    def aggregate_function(streamed_outputs):
        aggregate_output = ""
        for stream in streamed_outputs:
            aggregate_output += stream['text']

        # Aggregate metrics to expose to the user
        # input_tokens = -1 # TBD
        # output_tokens = -1 # TBD

        return {
            "text": aggregate_output,
            # "input_tokens": input_tokens,
            # "output_tokens": output_tokens,
        }

    # Streaming case
    for response in client.generate_stream(prompt, **sampling_params):
        if not response.token.special:
            text_outputs = response.token.text
            ret = {"text": text_outputs}

            # Update the aggregate transformation function
            runpod.serverless.modules.rp_metrics.metrics_collector.update_stream_aggregate(
                job_id=job['id'], 
                aggregate_function=aggregate_function
            )

            yield ret

    # Remove job from the set.
    JOBS.remove(job['id'])

# Start the serverless worker with appropriate settings
print("Starting the TGI serverless worker with streaming enabled.")
runpod.serverless.start({
    "handler": handler_streaming, 
    "concurrency_controller": concurrency_controller, 
    "return_aggregate_stream": True
})
