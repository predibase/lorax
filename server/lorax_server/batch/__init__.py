import grpc
from datasets import load_dataset
from transformers import AutoTokenizer

from lorax_server.pb import generate_pb2, generate_pb2_grpc


def run(
    input_path: str,
    output_path: str,
    tokenizer_name: str,
    prompt_column: str,
    input_format: str,
    uds_path: str,
):
    with grpc.insecure_channel("unix:///tmp/lorax-server-0") as channel:
        stub = generate_pb2_grpc.LoraxServiceStub(channel)

        # health check to ensure system is up
        resp = stub.Health(generate_pb2.HealthRequest())
        print("HEALTH RESPONSE", resp, type(resp))

        # warmup
        # TODO(travis): set warmup constraints based on input data

        # stream in the input file
        # TODO(travis): consider enabling streaming
        data_files = {"infer": input_path}
        dataset = load_dataset(input_format, data_files=data_files, split="infer", streaming=False)
        print(next(iter(dataset)))

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenized_dataset = dataset.map(
            lambda examples: tokenizer(examples[prompt_column], return_tensors="np"),
            batched=True,
        )

        print(tokenized_dataset[0])

        # stream out the output parquet file

