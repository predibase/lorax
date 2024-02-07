"""Script for testing dynamic adapter loading.

The script is relatively straightforward. It sends a blast of requests that
specify various valid/invalid adapter IDs to the server and records the time
taken for each request. The script then prints the average time taken for each
adapter ID. If the adapter ID is invalid, the server should return an error, 
in which case the time taken is -inf.

At the end of the script, the average time taken for each adapter ID is printed,
so invalid adapter IDs should have a time of -inf.

The script is run with the following command from the root directory:

```
make server-dev  # in terminal 1
make router-dev  # in terminal 2
python integration-tests/scripts/test_dynamic_adapter_loading.py  # in terminal 3
```

Here is a sample output of the script at time of writing. The average time taken
per adapter will vary based on the order/rate of requests hitting the server.

{
    'None': 13.32627587482847, 
    'a': -inf, 
    'arnavgrg/codealpaca-qlora': 5.320686455430655, 
    'justinxzhao/50451': -inf, 
    'justinxzhao/51318': 10.060468784400395, 
    'kashif/llama-7b_stack-exchange_RM_peft-adapter-merged': -inf, 
    'AdapterHub/xmod-base-zh_TW': -inf, 
    'b': -inf, 
    'c': -inf
}
"""

import collections
import concurrent.futures
import json
import random
import time
from urllib.request import Request, urlopen

import numpy as np


def query_lorax(args):
    prompt, adapter_id = args
    start_t = time.time()
    request_params = {
        "max_new_tokens": 128,
        "temperature": None,
        "details": True,
    }
    if adapter_id is not None:
        request_params["adapter_source"] = "local"
        request_params["adapter_id"] = adapter_id
        
    print("request_params", request_params)    
    url = "http://localhost:8080/generate"
    headers = {
        "Content-Type": "application/json",
    }
    data = json.dumps(
        {
            "inputs": prompt,
            "parameters": request_params,
        },
    ).encode("utf-8")
    request = Request(url, headers=headers, data=data)
    
    try:
        with urlopen(request) as response:
            response_body = json.loads(response.read().decode("utf-8"))
            ntokens = response_body["details"]["generated_tokens"]
            duration_s = time.time() - start_t
            # print(adapter_id, response_body["generated_text"])
    except Exception:
        print(f"exception in request: {adapter_id}")
        return adapter_id, 0, None

    print("adapter_id: {}\nCompleted {} in {:3f} seconds ({:3f} tokens / s)\n----".format(
        adapter_id,
        ntokens,
        duration_s,
        (ntokens / duration_s),
    ))
    return adapter_id, ntokens, duration_s, response_body["generated_text"]


def get_local_path(model_id):
    model_id = model_id.replace("/", "--")
    return f"/data/models--{model_id}/snapshots/834b33af35ff5965ea3e4bc18b51ad5d65da7466"



def main():
    prompt = """
Below is an instruction that describes a task, paired with an input
that provides further context. Write a response that appropriately
completes the request.

### Instruction: Write a function to return the first n fibonacci numbers

### Input: 

### Response:
"""
    NUM_REQUESTS = 500
    # N = 0
    # adapters = [get_local_path("arnavgrg/codealpaca_v3")] + [
    #     get_local_path(f"arnavgrg/codealpaca_v3_{i}")
    #     for i in range(1, N)
    # ]

    # Mistral
    # prompt = "[INST] Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? [/INST]"
    # adapters = [
    #     "vineetsharma/qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k",
    # ]
    
    # GPT2
    prompt = "Brand Name : First Aid Beauty ; Product Name : Ultra Repair Cream Intense Hydration ; Review Title :"
    adapters = ["/data/adapters/9789adb7-cd03-4862-91d5-b41b6746682e_ludwig/model_weights"]

    adapters += [None]
    # adapters = [None]

    # adapters += [
    # #     get_local_path("arnavgrg/codealpaca_v3"),
    # #     get_local_path("arnavgrg/codealpaca_v3_1"),
    # #     get_local_path("arnavgrg/codealpaca_v3_2"),
    # #     get_local_path("arnavgrg/codealpaca_v3_3"),
    # #     get_local_path("arnavgrg/codealpaca_v3_4"),
    # #     get_local_path("arnavgrg/codealpaca_v3_5"),
    # #     get_local_path("arnavgrg/codealpaca_v3_6"),
    # #     get_local_path("arnavgrg/codealpaca_v3_7"),
    # #     # get_local_path("arnavgrg/codealpaca_v3_8"),
    # #     # get_local_path("arnavgrg/codealpaca_v3_9"),

    # #     # valid
    # #     # "arnavgrg/codealpaca-qlora",
    # #     # "arnavgrg/codealpaca-qlora-v2",
    # #     # "arnavgrg/ludwig-webinar",
    # #     # "arnavgrg/ludwig-webinar-1",
    # #     # "arnavgrg/codealpaca_v3",
    # #     # "arnavgrg/codealpaca_v3_1",
    # #     # "AbhishekkV19/llama2-code-ludwig",
    # #     # "daochf/LudwigLlama2-PuceDS-v01",
    # #     # "hessertaboada/ludwig-webinar",
    # #     # "AmlanSamanta/ludwig-webinar",


    # #     # None,

    # #     # # download error: bad adapter name
    #     "abc",

    # #     # # download error: NaN weights
    #     "justinxzhao/50451",

    # #     # # download error: not an adapter
    #     "kashif/llama-7b_stack-exchange_RM_peft-adapter-merged",

    # #     # # load error: wrong base model
    #     "AdapterHub/xmod-base-zh_TW",
    # ]

    args_list = []
    for i in range(NUM_REQUESTS):
        adapter_id = adapters[i % len(adapters)]
        args_list.append((prompt, adapter_id))

    start_t = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        results = executor.map(query_lorax, args_list)
    span_s = time.time() - start_t

    total_tokens = 0
    total_duration_s = 0
    responses = collections.defaultdict(set)
    for adapter_id, ntokens, duration_s, resp in results:
        if duration_s is None:
            continue
        total_tokens += ntokens
        total_duration_s += duration_s
        responses[adapter_id].add(resp)

    print(f"Avg Latency: {total_duration_s / total_tokens} s / tokens")
    print(f"Throughput: {total_tokens / span_s} tokens / s")

    for adapter_id, resp in responses.items():
        print("----")
        print(f"{adapter_id}: {len(resp)}")
        for r in resp:
            print("    * " + r)
        print("----")

    # d = collections.defaultdict(list)
    # for adapter_id, ntokens, duration_s in results:
    #     d[str(adapter_id)].append(end_t)
    # print({k: np.mean(v) for k, v in d.items()})


if __name__ == '__main__':
    main()

