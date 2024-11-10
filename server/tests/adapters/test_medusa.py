import torch

from lorax_server.adapters.medusa import BatchMedusaWeights, MedusaConfig
from lorax_server.adapters.utils import download_adapter_weights
from lorax_server.adapters.weights import AdapterBatchMetadata
from lorax_server.models.causal_lm import CausalLM
from lorax_server.utils.adapter import load_module_map
from lorax_server.utils.lora import LM_HEAD
from lorax_server.utils.sources import HUB

model_id = "mistralai/Mistral-7B-Instruct-v0.2"
adapter_id = "predibase/Mistral-7B-Instruct-v0.2-medusa"


def test_batched_medusa_weights(default_causal_lm: CausalLM):
    download_adapter_weights(adapter_id, HUB)

    module_map, medusa_config, _, _ = load_module_map(
        model_id, adapter_id, HUB, tuple(), None
    )
    assert isinstance(medusa_config, MedusaConfig)

    medusa_weights = medusa_config.load_batched_adapter_weights(
        default_causal_lm,
        module_map,
        LM_HEAD,
        set(),
        False,
    )

    meta = AdapterBatchMetadata(
        adapter_indices=torch.tensor([0, 0, 1, 1, 0, 0, 1, 1], dtype=torch.int64),
        adapter_list=[0, 1, 0, 1],
        adapter_set={0, 1},
        adapter_segments=torch.tensor([0, 2, 4, 6, 8], dtype=torch.int64),
        segment_indices=[0, 1, 0, 1],
    )

    batch_medusa_weights = BatchMedusaWeights.load(
        {
            0: medusa_weights,
            1: medusa_weights,
        },
        meta,
        layer_name=LM_HEAD,
        prefill=False,
        prefill_head_indices=None,
    )

    assert batch_medusa_weights is not None
    assert batch_medusa_weights.default_medusa == medusa_weights
