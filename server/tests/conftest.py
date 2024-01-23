import pytest

from transformers import AutoTokenizer

from lorax_server.pb import generate_pb2
from lorax_server.models.causal_lm import CausalLM, CausalLMBatch
from lorax_server.utils.lora import AdapterBatchData
from lorax_server.utils.tokenizer import TokenizerManager
from lorax_server.utils.tokens import NextTokenChooser


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


@pytest.fixture(scope="session")
def default_causal_lm():
    return CausalLM("gpt2")


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
        torch.float32,
        torch.device("cpu"),
    )


# check generations work normally with temperature = 0
def test_generate_token_temperature_zero(default_causal_lm, default_causal_lm_batch):
    sequence_length = len(default_causal_lm_batch.all_input_ids[0])
    batch = default_causal_lm_batch

    # set all token choosers in batch to be deterministic with Temperature = 0
    determ_token_choosers = [
        NextTokenChooser(temperature=0) for _ in range(len(batch.next_token_choosers))
    ]
    batch.next_token_choosers = determ_token_choosers
    # generate tokens from next batch
    generations, next_batch = default_causal_lm.generate_token(default_causal_lm_batch)

    # same assertions as testing generate token, causal lm
    assert len(generations) == len(next_batch)
    assert isinstance(next_batch, CausalLMBatch)

    assert len(next_batch.all_input_ids) == len(next_batch)
    assert len(next_batch.all_input_ids[0]) == sequence_length + 1


# generates tokens with determinstic choosers,
# checks that output tokens have highest probability in distribution
def test_deterministic_tokens_temperature_zero(
    default_causal_lm, default_causal_lm_batch
):
    # Inside of CausalLM.generate_token, used to access
    # logit distribution and compare log prob
    batch = default_causal_lm_batch

    # set all token choosers in batch to be deterministic with Temperature = 0
    determ_token_choosers = [
        NextTokenChooser(temperature=0) for _ in range(len(batch.next_token_choosers))
    ]
    batch.next_token_choosers = determ_token_choosers

    attention_mask = batch.attention_mask[:, : -batch.padding_right_offset]

    adapter_data = AdapterBatchData.from_meta(
        batch.adapter_meta, default_causal_lm.batched_lora_weights
    )

    logits, _ = default_causal_lm.forward(
        batch.input_ids,
        attention_mask,
        batch.position_ids,
        batch.past_key_values,
        adapter_data,
    )

    # Zipped iterator
    iterator = zip(
        batch.requests,
        batch.input_lengths,
        batch.prefix_offsets,
        batch.read_offsets,
        logits,
        batch.next_token_choosers,
        batch.stopping_criterias,
        batch.all_input_ids,
    )

    # For each member of the batch
    for i, (
        request,
        input_length,
        prefix_offset,
        read_offset,
        logits,
        next_token_chooser,
        stopping_criteria,
        all_input_ids,
    ) in enumerate(iterator):
        # Select next token
        next_token_id, logprobs = next_token_chooser(
            all_input_ids.view(1, -1), logits[-1:, :]
        )

        # Generated token
        next_token_logprob = logprobs[-1, next_token_id]

        # A deterministic model with Temperature = 0 should always choose
        # the highest logprob token
        assert next_token_logprob == max(logprobs[-1])
