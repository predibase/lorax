import pytest
import torch
from transformers import AutoTokenizer

from lorax_server.models.causal_lm import CausalLM, CausalLMBatch
from lorax_server.pb import generate_pb2
from lorax_server.utils.dist import FakeGroup
from lorax_server.utils.tokenizer import TokenizerManager


@pytest.fixture
def default_pb_parameters():
    return generate_pb2.NextTokenChooserParameters(
        temperature=1.0,
        repetition_penalty=1.0,
        top_k=0,
        top_p=1.0,
        typical_p=1.0,
        do_sample=False,
    )


@pytest.fixture
def default_pb_stop_parameters():
    return generate_pb2.StoppingCriteriaParameters(stop_sequences=[], max_new_tokens=10)


@pytest.fixture
def default_json_schema():
    return """{
    "$defs": {
        "Armor": {
            "enum": ["leather", "chainmail", "plate"],
            "title": "Armor",
            "type": "string"
        }
    },
    "properties": {
        "name": {"maxLength": 10, "title": "Name", "type": "string"},
        "age": {"title": "Age", "type": "integer"},
        "armor": {"$ref": "#/$defs/Armor"},
        "strength": {"title": "Strength", "type": "integer"}\
    },
    "required": ["name", "age", "armor", "strength"],
    "title": "Character",
    "type": "object"
}"""


@pytest.fixture
def schema_constrained_pb_parameters(default_json_schema):
    return generate_pb2.NextTokenChooserParameters(
        temperature=1.0,
        repetition_penalty=1.0,
        top_k=0,
        top_p=1.0,
        typical_p=1.0,
        do_sample=False,
        schema=default_json_schema,
    )


@pytest.fixture(scope="session")
def default_causal_lm():
    model = CausalLM("gpt2")
    model.process_group = FakeGroup(0, 1)
    return model


@pytest.fixture(scope="session")
def gpt2_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="left")
    tokenizer.pad_token_id = 50256
    return tokenizer


@pytest.fixture
def default_pb_request(default_pb_parameters, default_pb_stop_parameters):
    return generate_pb2.Request(
        id=0,
        inputs="Test",
        prefill_logprobs=True,
        truncate=100,
        parameters=default_pb_parameters,
        stopping_parameters=default_pb_stop_parameters,
    )


@pytest.fixture
def default_pb_batch(default_pb_request):
    return generate_pb2.Batch(id=0, requests=[default_pb_request], size=1)


@pytest.fixture
def default_causal_lm_batch(default_pb_batch, gpt2_tokenizer):
    return CausalLMBatch.from_pb(
        default_pb_batch,
        gpt2_tokenizer,
        TokenizerManager(),
        None,
        None,
        torch.float32,
        torch.device("cpu"),
    )
