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


def query_tgi(args):
    prompt, adapter_id = args
    start_t = time.time()
    request_params = {
        "max_new_tokens": 100,
        "temperature": 0.01,
    }
    if adapter_id is not None:
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
            end_t = time.time() - start_t
    except Exception as e:
        end_t = float('-inf')

    print("adapter_id: {}\nCompleted in {:3f} seconds\n----".format(
        adapter_id, 
        end_t
    ))
    return adapter_id, end_t



def main():
    prompt = """
Below is an instruction that describes a task, paired with an input
that provides further context. Write a response that appropriately
completes the request.

### Instruction: Write a function to return the first n fibonacci numbers

### Input: 

### Response:
"""
    args_list = []
    for i in range(200):
        if i % 7 == 1:
            # download error: bad adapter name
            adapter_id = random.choice("abc")
        elif i % 7 == 2:
            # valid
            adapter_id = "arnavgrg/codealpaca-qlora"
        elif i % 7 == 3:
            # download error: NaN weights
            adapter_id = "justinxzhao/50451" 
        elif i % 7 == 4:
            # valid
            adapter_id = "justinxzhao/51318"
        elif i % 7 == 5:
            # download error: not an adapter
            adapter_id = "kashif/llama-7b_stack-exchange_RM_peft-adapter-merged" 
        elif i % 7 == 6:
            # load error: wrong base model
            adapter_id = "AdapterHub/xmod-base-zh_TW" 
        else:
            # valid
            adapter_id = None
        args_list.append((prompt, adapter_id))

    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        results = executor.map(query_tgi, args_list)

    d = collections.defaultdict(list)
    for adapter_id, end_t in results:
        d[str(adapter_id)].append(end_t)

    print({k: np.mean(v) for k, v in d.items()})


if __name__ == '__main__':
    main()

