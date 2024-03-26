from abc import ABC, abstractclassmethod
from typing import Dict

from lorax_server.utils.lora import AdapterBatchMetadata


class AdapterWeights(ABC):
    @abstractclassmethod
    def get_batch_type(cls) -> "BatchAdapterWeights":
        pass


class BatchAdapterWeights(ABC):
    @abstractclassmethod
    def key(self) -> str:
        pass

    @abstractclassmethod
    def load(self, adapter_weights: Dict[int, AdapterWeights], meta: AdapterBatchMetadata) -> "BatchAdapterWeights":
        pass
