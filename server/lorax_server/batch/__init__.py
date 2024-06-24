from dataclasses import dataclass
from loguru import logger
import time
from typing import Dict, List, Optional, Tuple
import grpc
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from lorax_server.pb import generate_pb2, generate_pb2_grpc


BLOCK_SIZE = 16
_ID = 0


@dataclass
class Entry:
    request: generate_pb2.Request
    input_length: int


def run(
    input_path: str,
    output_path: str,
    max_input_length: int,
    max_batch_prefill_tokens: int,
    max_total_tokens: int,
    max_new_tokens: Optional[int],
    tokenizer_name: str,
    prompt_column: str,
    input_format: str,
    uds_path: str,
):
    t0 = time.time()
    with grpc.insecure_channel(f"unix://{uds_path}") as channel:
        client = generate_pb2_grpc.LoraxServiceStub(channel)

        # health check to ensure system is up
        resp = client.Health(generate_pb2.HealthRequest())
        # print("HEALTH RESPONSE", resp, type(resp))
        logger.info("Health check successful")

        # get deployment info
        info = client.Info(generate_pb2.InfoRequest())
        window_size = info.window_size

        # warmup
        # TODO(travis): set warmup constraints based on input data
        max_supported_total_tokens = warmup(
            client=client,
            max_input_length=max_input_length,
            max_batch_prefill_tokens=max_batch_prefill_tokens,
            max_total_tokens=max_total_tokens,
        )
        print("WARMUP COMPLETE", max_supported_total_tokens)
        logger.info(f"Warmup complete, max supported total tokens: {max_supported_total_tokens}")

        # stream in the input file
        # TODO(travis): consider enabling streaming
        data_files = {"infer": input_path}
        dataset = load_dataset(input_format, data_files=data_files, split="infer", streaming=False)
        # print(next(iter(dataset)))

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenized_dataset = dataset.map(
            lambda examples: tokenizer(examples[prompt_column], return_tensors="np"),
            batched=True,
        )
        # print(tokenized_dataset[0])

        # Convert dataset to entries
        entries = []
        for i, example in enumerate(tokenized_dataset):
            text = example["text"]
            input_ids = example["input_ids"]
            entry = create_entry(
                request_id=i,
                inputs=text,
                input_length=len(input_ids),
                max_input_length=max_input_length,
                max_total_tokens=max_total_tokens,
                max_new_tokens=max_new_tokens,
            )
            entries.append(entry)

            # TODO(debugging):
            if i == 10:
                break
        
        # continuous batching
        outputs = {}
        token_budget = max_supported_total_tokens
        with tqdm(total=len(entries)) as pbar:
            while batch := next_batch(entries, max_batch_prefill_tokens, token_budget, BLOCK_SIZE, window_size):
                # prefill
                # print("BATCH", batch)
                cached_batch, stopped_generations = prefill(client, batch)
                add_outputs(stopped_generations, outputs)
                pbar.update(len(stopped_generations))

                while cached_batch is not None:
                    batches = [cached_batch]

                    # prefill
                    while new_batch := next_batch(entries, max_batch_prefill_tokens, token_budget, BLOCK_SIZE, window_size):
                        new_cached_batch, new_stopped_generations = prefill(client, new_batch)
                        add_outputs(new_stopped_generations, outputs)
                        pbar.update(len(new_stopped_generations))
                        if new_cached_batch is not None:
                            batches.append(new_cached_batch)

                    # decode
                    cached_batch, stopped_generations = decode(client, batches)
                    add_outputs(stopped_generations, outputs)
                    pbar.update(len(stopped_generations))

        print(outputs)
        # stream out the output parquet file
        # TODO(travis) explore streaing writing: https://stackoverflow.com/questions/64791558/create-parquet-files-from-stream-in-python-in-memory-efficient-manner
    
    print("BATCH RUN COMPLETE", time.time() - t0)
    print("OUTPUTS", outputs)


def add_outputs(
    generations: List[generate_pb2.Generation],
    outputs: Dict[int, str],
):
    for generation in generations:
        outputs[generation.request_id] = generation.generated_text.text


def warmup(
    client: generate_pb2_grpc.LoraxServiceStub,
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
    resp = client.Warmup(request)

    return resp.max_supported_total_tokens


def prefill(
    client: generate_pb2_grpc.LoraxServiceStub,
    batch: generate_pb2.Batch,
) -> Tuple[generate_pb2.CachedBatch, List[generate_pb2.Generation]]:
    resp = client.Prefill(generate_pb2.PrefillRequest(batch=batch))
    return filter_batch(client, resp.batch, resp.generations)


def decode(
    client: generate_pb2_grpc.LoraxServiceStub,
    batches: List[generate_pb2.CachedBatch],
) -> Tuple[generate_pb2.CachedBatch, List[generate_pb2.Generation]]:
    resp = client.Decode(generate_pb2.DecodeRequest(batches=batches))
    return filter_batch(client, resp.batch, resp.generations)


def filter_batch(
    client: generate_pb2_grpc.LoraxServiceStub,
    batch: generate_pb2.CachedBatch,
    generations: List[generate_pb2.Generation],
) -> Tuple[Optional[generate_pb2.CachedBatch], List[generate_pb2.Generation]]:
    stopped_generations = [
        generation for generation in generations
        if generation.generated_text is not None
    ]

    # Retain only requests that are still in entries
    stopped_request_ids = set(generation.request_id for generation in stopped_generations)
    del batch.request_ids[:]
    batch.request_ids.extend([id for id in batch.request_ids if id not in stopped_request_ids])

    if len(batch.request_ids) == 0:
        # All requests have been filtered out
        # Next batch is now empty
        # Clear it from the Python shards cache
        # We unwrap here as we need to panic since we cannot recover if this method fails
        client.ClearCache(generate_pb2.ClearCacheRequest(id=batch.id))
        return None, []
    
    # Filter Python shard cache
    # We unwrap here as we need to panic since we cannot recover if this method fails
    resp = client.FilterBatch(generate_pb2.FilterBatchRequest(batch_id=batch.id, request_ids=batch.request_ids))
    return resp.batch, stopped_generations


def create_entry(
    request_id: int,
    inputs: str,
    input_length: int,
    max_input_length: int,
    max_total_tokens: int,
    max_new_tokens: Optional[int],
) -> Entry:
    # We truncate the input on the server side to be sure that it has the correct size
    effective_max_new_tokens = max_new_tokens or (max_total_tokens - input_length)
    # print("max_new_tokens", effective_max_new_tokens, max_total_tokens, input_length)
    if input_length > max_input_length:
        raise ValueError(f"Input length {input_length} is greater than max_input_length {max_input_length}")

    request = generate_pb2.Request(
        id=request_id,
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
            repetition_penalty=1,
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

    return Entry(
        request=request,
        input_length=input_length,
    )


def next_batch(
    entries: List[Entry],
    max_batch_prefill_tokens: int,
    token_budget: int,
    block_size: int,
    window_size: Optional[int],
) -> Optional[generate_pb2.Batch]:
    batch_requests = []
    prefill_tokens = 0
    decode_tokens = 0
    while entries and (prefill_tokens + decode_tokens) <= token_budget:
        entry = entries.pop(0)

        # update prefill tokens
        prefill_tokens += int(((entry.input_length + block_size - 1) / block_size) * block_size)

        # update decode tokens
        if not window_size:
            max_new_tokens = entry.request.stopping_parameters.max_new_tokens
        else:
            max_new_tokens = min(
                window_size - entry.input_length,
                entry.request.stopping_parameters.max_new_tokens,
            )
        decode_tokens += int(((max_new_tokens + block_size - 1) / block_size) * block_size)

        if prefill_tokens > max_batch_prefill_tokens or (prefill_tokens + decode_tokens) > token_budget:
            # Entry is over budget
            # Add it back to the front
            entries.insert(0, entry)
            break

        batch_requests.append(entry.request)
    
    if not batch_requests:
        return None
    
    global _ID
    next_id = _ID
    _ID += 1

    # print(window_size, prefill_tokens, decode_tokens)

    return generate_pb2.Batch(
        id=next_id,
        size=len(batch_requests),
        requests=batch_requests,
        max_tokens=prefill_tokens + decode_tokens,
    )
