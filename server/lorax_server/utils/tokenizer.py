from typing import Optional

from transformers import PreTrainedTokenizerBase

from lorax_server.pb import generate_pb2


class TokenizerManager:
    def __init__(self):
        self.tokenizers = {}

    def add_tokenizer(self, adapter_idx: int, tokenizer: PreTrainedTokenizerBase):
        self.tokenizers[adapter_idx] = tokenizer

    def get_tokenizer(self, adapter_idx: int, default: PreTrainedTokenizerBase) -> Optional[PreTrainedTokenizerBase]:
        return self.tokenizers.get(adapter_idx, default)

    def get_inputs(
        self,
        r: generate_pb2.Request,
        base_tokenizer: PreTrainedTokenizerBase,
    ) -> str:
        return r.inputs
