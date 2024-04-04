import grpc
from lorax_server.pb import generate_pb2_grpc, generate_pb2
from google.protobuf import json_format

def run_prefil(stub):
    json_string = '''
    {
    "batch": {
        "requests": [
        {
            "inputs": "[INST] Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? [/INST]",
            "truncate": 1792,
            "parameters": {
            "temperature": 1,
            "top_p": 1,
            "typical_p": 1,
            "seed": 11242005690274133440,
            "repetition_penalty": 1
            },
            "stopping_parameters": {
            "max_new_tokens": 64
            },
            "adapter_index": 1
        }
        ],
        "size": 1,
        "max_tokens": 112
    }
    }
    '''

    request = generate_pb2.PrefillRequest()
    json_format.Parse(json_string, request)
    response = stub.Prefill(request)
    return response

def run_embed(stub):
    json_string = '''
    {
        "inputs": "[INST] Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? [/INST]"
    }
    '''
    request = generate_pb2.EmbedRequest()
    json_format.Parse(json_string, request)
    response = stub.Embed(request)
    return response


def run():
    # Connect to the server using a Unix domain socket
    channel = grpc.insecure_channel('unix:///tmp/lorax-server-0')

    # Create a stub (client)
    stub = generate_pb2_grpc.LoraxServiceStub(channel)
    embed_resp = run_embed(stub)
    breakpoint()
    prefil_resp = run_prefil(stub)
    batch = prefil_resp.batch

    # # Create a request object
    request = generate_pb2.DecodeRequest(
        batches=[batch]
    )

    # Call the Decode method
    for _ in range(100):
        try:
            response = stub.Decode(request)
            print("Client received: ", response)
        except grpc.RpcError as e:
            print(f"RPC failed: {e.code()} {e.details()}")

if __name__ == '__main__':
    run()

