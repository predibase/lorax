from lorax_server.adapters.weights import AdapterBatchData
from lorax_server.models.causal_lm import CausalLMBatch
from lorax_server.utils.tokens import (
    FinishReason,
    NextTokenChooser,
    StoppingCriteria,
    StopSequenceCriteria,
)


def test_stop_sequence_criteria():
    criteria = StopSequenceCriteria("/test;")

    assert not criteria("/")
    assert not criteria("/test")
    assert criteria("/test;")
    assert not criteria("/test; ")


def test_stop_sequence_criteria_escape():
    criteria = StopSequenceCriteria("<|stop|>")

    assert not criteria("<")
    assert not criteria("<|stop")
    assert criteria("<|stop|>")
    assert not criteria("<|stop|> ")


def test_stopping_criteria():
    criteria = StoppingCriteria(0, [StopSequenceCriteria("/test;")], max_new_tokens=5)
    assert criteria(65827, "/test") == (False, None)
    assert criteria(30, ";") == (True, FinishReason.FINISH_REASON_STOP_SEQUENCE)


def test_stopping_criteria_eos():
    criteria = StoppingCriteria(0, [StopSequenceCriteria("/test;")], max_new_tokens=5)
    assert criteria(1, "") == (False, None)
    assert criteria(0, "") == (True, FinishReason.FINISH_REASON_EOS_TOKEN)


def test_stopping_criteria_max():
    criteria = StoppingCriteria(0, [StopSequenceCriteria("/test;")], max_new_tokens=5)
    assert criteria(1, "") == (False, None)
    assert criteria(1, "") == (False, None)
    assert criteria(1, "") == (False, None)
    assert criteria(1, "") == (False, None)
    assert criteria(1, "") == (True, FinishReason.FINISH_REASON_LENGTH)


# check generations work normally with temperature = 0
def test_generate_token_temperature_zero(default_causal_lm, default_causal_lm_batch):
    sequence_length = len(default_causal_lm_batch.all_input_ids[0])
    batch = default_causal_lm_batch

    # set all token choosers in batch to be deterministic with Temperature = 0
    determ_token_choosers = [NextTokenChooser(temperature=0) for _ in range(len(batch.next_token_choosers))]
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
def test_deterministic_tokens_temperature_zero(default_causal_lm, default_causal_lm_batch):
    # Inside of CausalLM.generate_token, used to access
    # logit distribution and compare log prob
    batch = default_causal_lm_batch

    # set all token choosers in batch to be deterministic with Temperature = 0
    determ_token_choosers = [NextTokenChooser(temperature=0) for _ in range(len(batch.next_token_choosers))]
    batch.next_token_choosers = determ_token_choosers

    attention_mask = batch.attention_mask[:, : -batch.padding_right_offset]

    adapter_data = AdapterBatchData.from_meta(
        meta=batch.adapter_meta, 
        weights=default_causal_lm.layer_to_adapter_weights, 
        layer_to_lora_weights={},
        punica_wrapper=None,
        prefill=True, 
        prefill_head_indices=None,
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
        next_token_id, logprobs = next_token_chooser(all_input_ids.view(1, -1), logits[-1:, :])

        # Generated token
        next_token_logprob = logprobs[-1, next_token_id]

        # A deterministic model with Temperature = 0 should always choose
        # the highest logprob token
        assert next_token_logprob == max(logprobs[-1])
