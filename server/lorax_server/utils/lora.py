from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Type

import torch

from lorax_server.adapters.weights import AdapterWeights, BatchAdapterWeights


# Constants
Q_PROJ = "q_proj"
K_PROJ = "k_proj"
V_PROJ = "v_proj"
O_PROJ = "o_proj"

GATE_PROJ = "gate_proj"
UP_PROJ = "up_proj"
DOWN_PROJ = "down_proj"

LM_HEAD = "lm_head"
