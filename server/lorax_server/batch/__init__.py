import time
import grpc
from datasets import load_dataset
from transformers import AutoTokenizer

from lorax_server.pb import generate_pb2, generate_pb2_grpc


def run(
    input_path: str,
    output_path: str,
    max_input_length: int,
    max_batch_prefill_tokens: int,
    max_total_tokens: int,
    tokenizer_name: str,
    prompt_column: str,
    input_format: str,
    uds_path: str,
):
    t0 = time.time()
    with grpc.insecure_channel("unix:///tmp/lorax-server-0") as channel:
        stub = generate_pb2_grpc.LoraxServiceStub(channel)

        # health check to ensure system is up
        resp = stub.Health(generate_pb2.HealthRequest())
        print("HEALTH RESPONSE", resp, type(resp))

        # warmup
        # TODO(travis): set warmup constraints based on input data
        max_supported_total_tokens = warmup(
            stub=stub,
            max_input_length=max_input_length,
            max_batch_prefill_tokens=max_batch_prefill_tokens,
            max_total_tokens=max_total_tokens,
        )
        print("WARMUP COMPLETE", max_supported_total_tokens)

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
        # TODO(travis) explore streaing writing: https://stackoverflow.com/questions/64791558/create-parquet-files-from-stream-in-python-in-memory-efficient-manner
    
    print("BATCH RUN COMPLETE", time.time() - t0)


def warmup(
    stub: generate_pb2_grpc.LoraxServiceStub,
    max_input_length: int,
    max_batch_prefill_tokens: int,
    max_total_tokens: int,
) -> int:
    n_tokens = 0
    requests = []

    while n_tokens < max_batch_prefill_tokens:
        # We truncate the input on the server side to be sure that it has the correct size
        truncate_length = min(max_input_length, max_batch_prefill_tokens - n_tokens);
        requests.append(generate_pb2.Request(
            id=0,
            inputs="_test " * max_input_length,
            truncate=truncate_length,
            # Set sampling parameters to also take these ops into account in the max memory
            parameters=generate_pb2.NextTokenChooserParameters(
                temperature=0.9,
                top_k=10,
                top_p=0.9,
                typical_p=0.9,
                do_sample=False,
                seed=0,
                repetition_penalty=1.2,
                watermark=True,
                adapter_id="",
                schema=None,
                return_k_alternatives=0,
            ),
            stopping_parameters=generate_pb2.StoppingCriteriaParameters(
                max_new_tokens=max_total_tokens - truncate_length,
                stop_sequences=[],
                ignore_eos_token=False,
            ),
            adapter_index=0,
            prefill_logprobs=True,
            apply_chat_template=False,
        ))
        n_tokens += max_input_length

    batch = generate_pb2.Batch(
        id=0,
        size=len(requests),
        requests=requests,
        max_tokens=0,
    )

    max_new_tokens = max_total_tokens - max_input_length;
    request = generate_pb2.WarmupRequest(
        batch=batch,
        max_new_tokens=max_new_tokens,
    )
    resp = stub.Warmup(request)

    return resp.max_supported_total_tokens
