import json
from pathlib import Path
import pickle

import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx.core import transpose
import safetensors
from safetensors.torch import save_file
import torch

import mlx.core as mx

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


def get_peft_model(
    model, linear_layers, rank=10, alpha=0.1, dropout=0.1, freeze=True, verbose=True
):
    if freeze:
        freeze_model(model)

    for name, module in model.language_model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.QuantizedLinear):
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


def freeze_model(model):
    for name, module in model.named_modules():
        name = name.split(".")[0]
        if name in [
            "language_model",
            "vision_model",
            "vision_tower",
            "aligner",
            "connector",
            "multi_modal_projector",
            "mm_projector",
        ]:
            model[f"{name}"].freeze()


def find_all_linear_names(model):
    cls = nn.Linear
    quantized_cls = nn.QuantizedLinear
    lora_module_names = set()
    multimodal_keywords = [
        "mm_projector",
        "vision_tower",
        "vision_resampler",
        "aligner",
    ]
    # print('ALL MODULES IN MODULE', model.named_modules())
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls) or isinstance(module, quantized_cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    # print('RESULTING MODULES', list(lora_module_names))
    return list(lora_module_names)


def count_parameters(model):
    def nparams(m):
        if isinstance(m, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
            return m.weight.size * (32 // m.bits)
        return sum(v.size for _, v in tree_flatten(m.parameters()))

    leaf_modules = tree_flatten(
        model.leaf_modules(), is_leaf=lambda m: isinstance(m, nn.Module)
    )
    total_p = sum(nparams(m) for _, m in leaf_modules) / 10**6

    return total_p


def print_trainable_parameters(model):
    def nparams(m):
        if isinstance(m, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
            return m.weight.size * (32 // m.bits)
        return sum(v.size for _, v in tree_flatten(m.parameters()))

    leaf_modules = tree_flatten(
        model.leaf_modules(), is_leaf=lambda m: isinstance(m, nn.Module)
    )
    total_p = sum(nparams(m) for _, m in leaf_modules) / 10**6
    trainable_p = (
        sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 10**6
    )

    print(
        f"#trainable params: {trainable_p} M || all params: {total_p} M || trainable%: {(trainable_p * 100 / total_p):.3f}%"
    )


def torch_to_mx(a: torch.Tensor, *, dtype: str) -> mx.array:
    # bfloat16 is not numpy convertible. Upcast to float32 to avoid precision loss
    a = a.to(torch.float32) if dtype == "bfloat16" else a.to(getattr(torch, dtype))
    return mx.array(a.numpy(), getattr(mx, dtype))

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

    # TODO: add lora params to the config and load them here
    list_of_modules = find_all_linear_names(model.language_model.model)
    if config is not None:
        model = get_peft_model(model, list_of_modules, **config)
    else:
        model = get_peft_model(model, list_of_modules)


    # TODO: remove debug lines
    tensors = {}
    dtype = "float16" # ["float16", "bfloat16", "float32"]
    # dtype = getattr(mx, dtype)
    with open('/Users/ngc436/Documents/projects/mlx-vlm/mlx_vlm/module_names_correspondance_dict.pkl', 'rb') as fp:
        module_names_dict = pickle.load(fp)
    with safetensors.safe_open(str(adapter_path / "adapters.safetensors"), framework="pt", device="cpu") as f:
        # print(f.keys())
        for key in f.keys():
            t_res = torch_to_mx(f.get_tensor(key), dtype=dtype)
            # t_res = f.get_tensor(key)
            tensors[module_names_dict[key]] = transpose(t_res) # 0, 1
            # tensors[module_names_dict[key]] = torch.transpose(t_res, 0, 1).contiguous()
    mx.save_safetensors(str(adapter_path / "adapters_v2.safetensors"), tensors)
    # safetensors.torch.save_file(tensors, str(adapter_path / "adapters_v2.safetensors"))

    # TODO: Use custom adapter name
    model.load_weights(str(adapter_path / "adapters_v2.safetensors"), strict=False)

    return model
