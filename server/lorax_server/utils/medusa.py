import torch
from dataclasses import dataclass
from lorax_server.utils.layers import FastLinear


@dataclass
class Output:
    logits: torch.FloatTensor = None
    speculative_logits: torch.FloatTensor = None


class ResBlock(torch.nn.Module):
    def __init__(self, config, prefix, weights):
        super().__init__()
        self.linear = FastLinear.load(
            config, prefix=f"{prefix}.linear", weights=weights, bias=True
        )
        self.act = torch.nn.SiLU()

    def forward(self, x):
        return x + self.act(self.linear(x))


class MedusaModel(torch.nn.Module):
    def __init__(self, config, weights, lm_head):
        super().__init__()

        self.heads = torch.nn.ModuleList()
        for i in range(config["medusa_num_heads"]):
            head = MedusaHead(config, prefix=f"{i}", weights=weights)
            self.heads.append(head)

        self.lm_head = lm_head

    def forward(self, x):
        logits = self.lm_head(x)
        speculative_logits = torch.stack([head(x) for head in self.heads], dim=1)
        return logits, speculative_logits


class MedusaHead(torch.nn.Module):
    def __init__(self, config, prefix, weights):
        super().__init__()

        self.blocks = torch.nn.ModuleList()
        for i in range(config["medusa_num_layers"]):
            block = ResBlock(config, prefix=f"{prefix}.{i}", weights=weights)
            self.blocks.append(block)

        n = len(self.blocks)
        self.out = FastLinear.load(
            config, prefix=f"{prefix}.{n}", weights=weights, bias=False
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.out(x)
        return x
