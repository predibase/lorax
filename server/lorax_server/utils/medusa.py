import torch
from dataclasses import dataclass
from lorax_server.utils.layers import FastLinear


@dataclass
class Output:
    logits: torch.FloatTensor = None
    speculative_logits: torch.FloatTensor = None


class ResBlock(torch.nn.Module):
    """
    Residual block module.

    Args:
        config (dict): Configuration for the block.
        prefix (str): Prefix for the block.
        weights (torch.Tensor): Weights for the block.

    Attributes:
        linear (FastLinear): Linear layer.
        act (torch.nn.SiLU): Activation function.

    """

    def __init__(self, config, prefix, weights):
        super().__init__()
        self.linear = FastLinear.load(
            config, prefix=f"{prefix}.linear", weights=weights, bias=True
        )
        self.act = torch.nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the residual block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        return x + self.act(self.linear(x))


class MedusaModel(torch.nn.Module):
    """
    MedusaModel is a PyTorch module that represents the Medusa model.

    Args:
        config (dict): Configuration parameters for the Medusa model.
        weights (list): List of weights for the Medusa model.
        lm_head (torch.nn.Module): Language model head for the Medusa model.

    Attributes:
        heads (torch.nn.ModuleList): List of MedusaHead modules.
        lm_head (torch.nn.Module): Language model head for the Medusa model.
    """

    def __init__(self, config, weights, lm_head):
        super().__init__()

        self.heads = torch.nn.ModuleList()
        for i in range(config["medusa_num_heads"]):
            head = MedusaHead(config, prefix=f"{i}", weights=weights)
            self.heads.append(head)

        self.lm_head = lm_head

    def forward(self, x):
        """
        Forward pass of the MedusaModel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: A tuple containing the logits and speculative logits.
        """
        logits = self.lm_head(x)
        speculative_logits = torch.stack([head(x) for head in self.heads], dim=1)
        return logits, speculative_logits


class MedusaHead(torch.nn.Module):
    """
    MedusaHead is a module that represents the head of the Medusa network.

    Args:
        config (dict): Configuration parameters for the Medusa network.
        prefix (str): Prefix for naming the layers of the MedusaHead module.
        weights (dict): Pretrained weights for the Medusa network.

    Attributes:
        blocks (torch.nn.ModuleList): List of ResBlock modules.
        out (FastLinear): Output layer of the MedusaHead module.
    """

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
