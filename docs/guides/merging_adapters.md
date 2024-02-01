# Merging Adapters

In LoRAX, multiple LoRA adapters can be merged together per request to create powerful multi-task ensembles
using one of several different [merge strategies](#merge-strategies).

This is particularly useful when you want your LLM to be capable of handling multiple types of tasks based on the user's prompt without
requiring them to specify the type of task they wish to perform.

## Background: Model Merging

Model merging is a set of techniques popularized by frameworks like [mergekit](https://github.com/cg123/mergekit) that allow taking
multiple specialized fine-tuned models and combining their weights together to output a single model that can perform each of these
tasks with a much smaller total footprint.

A common use case could be to train specialized LoRA adapters for tasks like SQL generation, customer support email
generation, and information extraction. Without model merging, the user submitting their query will need to know in advance which
of these models to route their query to. With model merging, the user should be able to submit their query without prior knowledge
of which backing adapter is best suited to respond to the query.

In some cases the mixing of adapter specializations could even result in a better final response. For example, by mixing an adapter that understand math with an adapter that can provide detailed and intuitive explanations, the user could in theory get correct answers to math questions with detailed step-by-step reasoning to aide in the user's learning.

## Merge Strategies

LoRAX provides a number of model merging methods taken from [mergekit](https://github.com/cg123/mergekit) and [PEFT](https://github.com/huggingface/peft).

Options:

- `linear` (default)
- `ties`
- `dare_linear`
- `dare_ties`

### Linear

The default and most straightforward way to merge model adapters is to linearly combine each of the parameters as a weighted average. This idea was 
explored in the context of merging fine-tuned models in [Model Soups](https://arxiv.org/abs/2203.05482).

Parameters:

- `weights` (default: `[1, ..]`): relative weight of each of the adapters in the request.

### TIES

[TIES](https://arxiv.org/abs/2306.01708) is based on the idea of [Task Arithmetic](https://arxiv.org/abs/2212.04089), whereby the fine-tuned models 
are merged after subtracting out the base model weights. LoRA and other adapters are already task-specific tensors, 
so this approach is a natural fit when merging LoRAs.

To resolve interference between adapters, the weights are sparsified and a sign-based consensus algorithms is used to determine the weighted average.

One the strengths of this approach is its ability to scale well to large numbers of adapters and retain each of their strengths.

Parameters:

- `weights` (default: `[1, ..]`): relative weight of each of the adapters in the request.
- `density` (required): fraction of weights in adapters to retain.
- `majority_sign_method` (default: `total`): one of `{total, frequency}` used to obtain the magnitude of the sign for consensus.

### DARE (Linear)

[DARE](https://arxiv.org/abs/2311.03099), like TIES, sparsifies adapter weights (task vectors) to reduce interference. Unlike TIES, however,
DARE uses random pruning and rescaling in an attempt to better match performance of the independent adapters.

Parameters:

- `weights` (default: `[1, ..]`): relative weight of each of the adapters in the request.
- `density` (required): fraction of weights in adapters to retain.

### DARE (TIES)

DARE method from above that also applies the sign consensus algorithm from TIES.

Parameters:

- `weights` (default: `[1, ..]`): relative weight of each of the adapters in the request.
- `density` (required): fraction of weights in adapters to retain.
- `majority_sign_method` (default: `total`): one of `{total, frequency}` used to obtain the magnitude of the sign for consensus.

## Example

This example is derived from the [PEFT example](https://github.com/huggingface/peft/blob/smangrul/add-new-merging-methods/examples/multi_adapter_examples/Lora_Merging.ipynb) for model merging.

First deploy LoRAX using the base model `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`, then run the following using
the [LoRAX Python Client](../reference/python_client.md):

```python
from lorax import Client, MergedAdapters

client = Client(endpoint_url)

# tinyllama merge
merged_adapters = MergedAdapters(
    ids=[
        "smangrul/tinyllama_lora_norobots",
        "smangrul/tinyllama_lora_sql",
        "smangrul/tinyllama_lora_adcopy",
    ],
    weights=[2.0, 0.3, 0.7],
    merge_strategy="ties",
    density=0.2,
    majority_sign_method="total",
)

# norobots
prompt = """<s><|im_start|>user
Write an essay about Generative AI.<|im_end|>
<|im_start|>assistant \n"""
response = client.generate(prompt, merged_adapters=merged_adapters)
print(response.generated_text)

# adcopy
prompt = """<s><|im_start|>system 
Create a text ad given the following product and description.<|im_end|> 
<|im_start|>user 
Product: Sony PS5 PlayStation Console
Description: The PS5™ console unleashes new gaming possibilities that you never anticipated.<|im_end|> 
<|im_start|>assistant \n"""
response = client.generate(prompt, merged_adapters=merged_adapters)
print(response.generated_text)

# sql
prompt = """<s> Table: 2-11365528-2
Columns: ['Team', 'Head Coach', 'President', 'Home Ground', 'Location']
Natural Query: Who is the Head Coach of the team whose President is Mario Volarevic?
SQL Query:"""
response = client.generate(prompt, merged_adapters=merged_adapters)
print(response.generated_text)
```
