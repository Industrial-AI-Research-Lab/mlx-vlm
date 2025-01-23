import json
from pathlib import Path
import pickle

import torch
import torch.nn as nn
from safetensors.torch import save_file

from .lora import LoRaLayer


def get_module_by_name(model, name):
    parts = name.split(".")
    module = model
    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def set_module_by_name(model, name, new_module):
    parts = name.split(".")
    module = model
    for part in parts[:-1]:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    if parts[-1].isdigit():
        module[int(parts[-1])] = new_module
    else:
        setattr(module, parts[-1], new_module)


def find_all_linear_names(model):
    """Find all linear layer names in the model."""
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.modules.linear.NonDynamicallyQuantizableLinear)):
            linear_layers.append(name.split(".")[-1])
    return linear_layers


def get_peft_model(
    model, linear_layers, rank=10, alpha=0.1, dropout=0.1, freeze=True, verbose=True
):
    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    for name, module in model.language_model.named_modules():
        if isinstance(module, (nn.Linear, nn.modules.linear.NonDynamicallyQuantizableLinear)):
            if name.split(".")[-1] in linear_layers:
                lora_layer = LoRaLayer(module, rank, alpha, dropout)
                set_module_by_name(model.language_model, name, lora_layer)

    model.config.lora = {}
    model.config.lora["rank"] = rank
    model.config.lora["alpha"] = alpha
    model.config.lora["dropout"] = dropout

    if verbose:
        print_trainable_parameters(model.language_model)

    return model


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def count_parameters(model):
    """
    Returns the number of parameters in the model.
    """
    return sum(p.numel() for p in model.parameters())


def apply_lora_layers(model: nn.Module, 
                     adapter_path: str,
                     adapter_type: str = 'mlx_vlm') -> nn.Module:
    """
    Apply LoRA layers to the model.

    Args:
        model (nn.Module): The neural network model.
        adapter_path (str): Path to the adapter configuration file.
        adapter_type (str): Type of adapter to use. Available options: 'mlx_vlm', 'peft', 'unsloth'
    Returns:
        nn.Module: The updated model with LoRA layers applied.
    """
    adapter_path = Path(adapter_path)

    if not adapter_path.exists():
        raise FileNotFoundError(f"The adapter path does not exist: {adapter_path}")

    # Check if the adapter has lora params in the config (adapter_config.json)
    with open(adapter_path / "adapter_config.json", "r") as f:
        config = json.load(f)
        if "rank" not in config:
            raise ValueError("The adapter does not have lora params in the config")

    # Apply LoRA layers
    list_of_modules = find_all_linear_names(model.language_model)
    if config is not None:
        model = get_peft_model(model, list_of_modules, **config)
    else:
        model = get_peft_model(model, list_of_modules)

    # Load adapter weights
    adapter_fname = "adapters.safetensors"
    if adapter_type == "mlx_vlm":
        state_dict = {}
        with open(adapter_path / "adapters.safetensors", "rb") as f:
            state_dict = torch.load(f)
        model.load_state_dict(state_dict, strict=False)

    return model
