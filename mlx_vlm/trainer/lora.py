import math
from typing import Union

import torch
import torch.nn as nn


class LoRaLayer(nn.Module):
    def __init__(
        self,
        linear: Union[nn.Linear, nn.modules.linear.NonDynamicallyQuantizableLinear],
        rank: int,
        alpha: float = 0.1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.original_layer = linear
        self.dropout = nn.Dropout(p=dropout)

        output_dims, input_dims = linear.weight.shape

        std_dev = 1 / math.sqrt(rank)

        self.A = nn.Parameter(
            torch.empty((input_dims, rank)).uniform_(-std_dev, std_dev)
        )
        self.B = nn.Parameter(torch.zeros((rank, output_dims)))
        self.alpha = alpha

    def forward(self, x):
        y = self.original_layer(x)
        lora_update = (self.dropout(x) @ self.A) @ self.B
        return y + (self.alpha * lora_update).to(x.dtype)


def replace_lora_with_linear(model):
    for name, module in model.named_modules():
        if isinstance(module, LoRaLayer):
            # Compute the final merged weight
            lora_update = module.alpha * (module.A @ module.B)
            updated_weight = module.original_layer.weight + lora_update.T
            use_bias = module.original_layer.bias is not None

            updated_bias = module.original_layer.bias

            # Create a new Linear layer with the updated parameters
            new_linear_layer = nn.Linear(
                updated_weight.size(1), updated_weight.size(0), bias=use_bias
            )

            new_linear_layer.weight = nn.Parameter(updated_weight)

            if use_bias:
                new_linear_layer.bias = nn.Parameter(updated_bias)

            # Replace the LoRaLayer with the new Linear layer in the model
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            if parent_name:
                parent = model.get_submodule(parent_name)
                setattr(parent, child_name, new_linear_layer)
            else:
                setattr(model, child_name, new_linear_layer)
