import json

import torch
from pydantic import BaseModel, constr
from transformers import AutoTokenizer

from lorax_server.utils.logits_process import OutlinesLogitsProcessor


class Person(BaseModel):
    name: constr(max_length=10)
    age: int


def test_outlines_process():
    torch.manual_seed(42)

    schema = json.dumps(Person.schema())
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    logit_processor = OutlinesLogitsProcessor(schema, tokenizer)

    B = 1
    V = tokenizer.vocab_size

    generated_tokens = []
    max_steps = 1000
    for step in range(max_steps):
        scores = torch.randn(B, V)
        biased_scores = logit_processor(scores)
        next_token_id = biased_scores.argmax(dim=-1).item()
        if next_token_id == tokenizer.eos_token_id:
            break

        generated_tokens.append(next_token_id)

        logit_processor.next_state(next_token_id)
        if logit_processor.fsm_state == -1:
            break

    if step == max_steps - 1:
        raise RuntimeError("Max steps reached")

    text = tokenizer.decode(generated_tokens)
    try:
        decoded_json = json.loads(text)
    except json.JSONDecodeError:
        raise RuntimeError(f"Failed to decode JSON: {text}")

    Person(**decoded_json)
