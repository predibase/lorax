from typing import Dict, Optional, TypeVar

import torch

from lorax_server.models.types import Batch

B = TypeVar("B", bound=Batch)


class Cache:
    """
    A class representing a cache.

    Attributes:
        cache (Dict[int, B]): A dictionary representing the cache, where the keys are batch IDs and the values are entries.

    Methods:
        pop(batch_id: int) -> Optional[B]: Removes and returns the entry with the specified batch ID from the cache.
        set(entry: B): Adds the specified entry to the cache.
        delete(batch_id: int): Deletes the entry with the specified batch ID from the cache.
        clear(): Clears the cache.
        __len__(): Returns the number of entries in the cache.
    """

    def __init__(self):
        self.cache: Dict[int, B] = {}

    def pop(self, batch_id: int) -> Optional[B]:
        return self.cache.pop(batch_id, None)

    def set(self, entry: B):
        if entry is not None:
            self.cache[entry.batch_id] = entry

    def delete(self, batch_id: int):
        batch = self.pop(batch_id)
        if batch is not None:
            del batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def clear(self):
        keys = list(self.cache.keys())
        for k in keys:
            self.delete(k)

    def __len__(self):
        return len(self.cache.keys())
