import json
from transformers import PreTrainedTokenizerBase

from lorax_server.pb import generate_pb2


def get_inputs(r: generate_pb2.Request, tokenizer: PreTrainedTokenizerBase) -> str:
    inputs = r.inputs
    if r.apply_chat_template:
        inputs = json.loads(inputs)
        inputs = tokenizer.apply_chat_template(inputs, add_generation_prompt=True, tokenize=False)
    return inputs
