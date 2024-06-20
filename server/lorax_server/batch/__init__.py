import time
from typing import List, Optional
import grpc
from datasets import load_dataset
from tqdm import tqdm
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
    with grpc.insecure_channel(f"unix://{uds_path}") as channel:
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

        # Convert dataset to requests
        requests = []
        for i, example in enumerate(tokenized_dataset):
                text = example["text"]
                input_ids = example["input_ids"]
                request = create_request(
                    inputs=text,
                    input_length=len(input_ids),
                    max_input_length=max_input_length,
                    max_total_tokens=max_total_tokens,
                )
                requests.append(request)

        # continuous batching
        cached_batch = None
        token_budget = max_supported_total_tokens
        with tqdm(total=len(tokenized_dataset)) as pbar:
            while requests:
                request = requests.pop(0)
            

                prefill_tokens += ((entry.request.input_length() + self.block_size - 1)
                    / self.block_size)
                    * self.block_size;

                batch = generate_pb2.Batch(
                    id=i,
                    size=1,
                    requests=[request],
                    max_tokens=0,
                )

                # prefill
                    

                pbar.update(1)

        # stream out the output parquet file
        # TODO(travis) explore streaing writing: https://stackoverflow.com/questions/64791558/create-parquet-files-from-stream-in-python-in-memory-efficient-manner
    
    print("BATCH RUN COMPLETE", time.time() - t0)


def next_batch(
    requests: List[generate_pb2.Request],
    max_batch_prefill_tokens: int,
    token_budget: int,
) -> Optional[generate_pb2.Batch]:
    batch_requests = []
    prefill_tokens = 0
    decode_tokens = 0
    while requests and max_batch_tokens <= token_budget:
        request = requests.pop(0)

        # pad to block size
        prefill_tokens += ((request.input_length + self.block_size - 1)
            / self.block_size)
            * self.block_size;


        batch_requests.append(request)
        max_batch_tokens += request.stopping_parameters.max_new_tokens
    
    return generate_pb2.Batch(
        id=0,
        size=len(batch_requests),
        requests=batch_requests,
        max_tokens=max_batch_tokens,
    )


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
        truncate_length = min(max_input_length, max_batch_prefill_tokens - n_tokens)
        requests.append(generate_pb2.Request(
            id=0,
            inputs="_test " * max_input_length,
            truncate=max_input_length,
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


def create_request(
    inputs: str,
    input_length: int,
    max_input_length: int,
    max_total_tokens: int,
) -> generate_pb2.Request:
    # We truncate the input on the server side to be sure that it has the correct size
    effective_max_new_tokens = max_total_tokens - input_length
    return generate_pb2.Request(
        id=0,
        inputs=inputs,
        truncate=max_input_length,
        # Set sampling parameters to also take these ops into account in the max memory
        parameters=generate_pb2.NextTokenChooserParameters(
            temperature=1,
            top_k=0,
            top_p=1,
            typical_p=1,
            do_sample=False,
            seed=0,
            repetition_penalty=1.2,
            watermark=False,
            adapter_id="",
            schema=None,
            return_k_alternatives=0,
        ),
        stopping_parameters=generate_pb2.StoppingCriteriaParameters(
            max_new_tokens=effective_max_new_tokens,
            stop_sequences=[],
            ignore_eos_token=False,
        ),
        adapter_index=0,
        prefill_logprobs=True,
        apply_chat_template=False,
    )
