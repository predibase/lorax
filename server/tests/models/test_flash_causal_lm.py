import torch
import pytest

from lorax_server.pb import generate_pb2
from lorax_server.models.flash_causal_lm import FlashCausalLMBatch


@pytest.fixture
def default_pb_batch(default_pb_request) -> generate_pb2.Batch:
    return generate_pb2.Batch(id=0, requests=[default_pb_request], size=1)


@pytest.fixture
def default_causal_lm_batch(default_pb_batch, gpt2_tokenizer) -> FlashCausalLMBatch:
    return FlashCausalLMBatch.from_pb(
        default_pb_batch, gpt2_tokenizer, torch.float32, torch.device("cpu")
    )


def test_batch_from_pb(default_pb_batch: generate_pb2.Batch, default_causal_lm_batch: FlashCausalLMBatch):
    batch = default_causal_lm_batch

    assert batch.batch_id == default_pb_batch.id
    assert batch.requests == default_pb_batch.requests

    assert len(batch.input_ids) == default_pb_batch.size
    assert batch.input_ids[0][-1] == 14402
    assert torch.all(batch.input_ids[0][:-1] == 50256)

    assert batch.attention_mask[0, 0] == 1
    assert torch.all(batch.attention_mask[0, 1:] == 0)

    assert batch.past_key_values is None

    assert all(
        [
            torch.equal(input_ids, all_input_ids[:, 0])
            for input_ids, all_input_ids in zip(batch.input_ids, batch.all_input_ids)
        ]
    )

    assert batch.input_lengths == [1]

    assert len(batch) == default_pb_batch.size
    assert len(batch.next_token_choosers) == len(batch.stopping_criterias) == len(batch)

    assert batch.max_input_length == batch.input_lengths[0]