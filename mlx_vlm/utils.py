import copy
import glob
import importlib
import json
import logging
import shutil
import time
import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import safetensors
import torch
import torch.nn as nn
import numpy as np
import requests
from huggingface_hub import snapshot_download, HfApi, ModelCard
from PIL import Image, ImageOps
from transformers import (
    AutoConfig,
    AutoProcessor,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from .models.base import BaseImageProcessor
from .trainer.utils import apply_lora_layers
from .sample_utils import top_p_sampling
from .tokenizer_utils import load_tokenizer

# Constants
MODEL_REMAPPING = {"llava-qwen2": "llava_bunny", "bunny-llama": "llava_bunny"}

MAX_FILE_SIZE_GB = 5


@dataclass
class GenerationResult:
    text: str
    token: Optional[int]
    logprobs: Optional[List[float]]
    prompt_tokens: int
    generation_tokens: int
    prompt_tps: float
    generation_tps: float
    peak_memory: float


def get_model_and_args(config: dict):
    """
    Retrieve the model object based on the configuration.

    Args:
        config (dict): The model configuration.

    Returns:
        A tuple containing the Model class and the ModelArgs class.
    """
    model_type = config["model_type"]
    model_type = MODEL_REMAPPING.get(model_type, model_type)
    try:
        arch = importlib.import_module(f"mlx_vlm.models.{model_type}")
    except ImportError:
        msg = f"Model type {model_type} not supported."
        logging.error(msg)
        raise ValueError(msg)

    return arch, model_type


def get_model_path(path_or_hf_repo: str, revision: Optional[str] = None) -> Path:
    """
    Ensures the model is available locally. If the path does not exist locally,
    it is downloaded from the Hugging Face Hub.

    Args:
        path_or_hf_repo (str): The local path or Hugging Face repository ID of the model.
        revision (str, optional): A revision id which can be a branch name, a tag, or a commit hash.

    Returns:
        Path: The path to the model.
    """
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        model_path = Path(
            snapshot_download(
                repo_id=path_or_hf_repo,
                revision=revision,
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.py",
                    "tokenizer.model",
                    "*.tiktoken",
                    "*.txt",
                ],
                resume_download=True,
            )
        )
    return model_path


def load_model(model_path: Path, lazy: bool = False, **kwargs) -> nn.Module:
    """
    Load and initialize the model from a given path.

    Args:
        model_path (Path): The path to load the model from.
        lazy (bool): If False eval the model parameters to make sure they are
            loaded in memory before returning, otherwise they will be loaded
            when needed. Default: ``False``

    Returns:
        nn.Module: The loaded and initialized model.

    Raises:
        FileNotFoundError: If the weight files (.safetensors) are not found.
        ValueError: If the model class or args class are not found or cannot be instantiated.
    """
    config = load_config(model_path, **kwargs)
    quantization = config.get("quantization", None)

    weight_files = glob.glob(str(model_path / "*.safetensors"))
    if not weight_files:
        logging.error(f"No safetensors found in {model_path}")
        message = f"""
No safetensors found in {model_path}
Create safetensors using the following code:
```
from transformers import AutoModelForCausalLM, AutoProcessor

model_id= "<huggingface_model_id>"
model = AutoModelForCausalLM.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

model.save_pretrained("<local_dir>")
processor.save_pretrained("<local_dir>")
```
Then use the <local_dir> as the --hf-path in the convert script.
```
python -m mlx_vlm.convert --hf-path <local_dir> --mlx-path <mlx_dir>
```
        """
        raise FileNotFoundError(message)

    weights = {}
    for wf in weight_files:
        print(wf)
        with safetensors.safe_open(wf, framework="pt", device="cpu") as f:
            for k in f.keys():
                weights[k] = f.get_tensor(k)

    model_class, model_type = get_model_and_args(config=config)

    # Initialize text and vision configs if not present
    config.setdefault("text_config", {})
    config.setdefault("vision_config", {})

    # Get vision config settings with defaults
    vision_config = config.get("vision_config", {})
    skip_vision = vision_config.get("skip_vision", False)

    # Initialize model config and update it with module configs
    model_config = model_class.ModelConfig.from_dict(config)
    modules = ["text", "vision", "perceiver", "projector"]
    model_config = update_module_configs(model_config, model_class, config, modules)

    model = model_class.Model(model_config)

    # Sanitize weights
    weights = sanitize_weights(model, weights)
    weights = sanitize_weights(
        model_class.VisionModel, weights, model_config.vision_config
    )
    weights = sanitize_weights(
        model_class.LanguageModel, weights, model_config.text_config
    )

    if (quantization := config.get("quantization", None)) is not None:
        # Handle legacy models which may not have everything quantized
        class_predicate = get_class_predicate(skip_vision, weights)
        
        # Convert MLX quantization to PyTorch quantization
        if hasattr(model, "to"):
            if quantization.get("bits", 0) == 8:
                model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
            elif quantization.get("bits", 0) == 4:
                # PyTorch doesn't support 4-bit quantization natively
                # You may need a custom quantization solution or third-party library
                pass

    # Convert weights to PyTorch tensors and load them
    state_dict = {k: torch.tensor(v.numpy()) for k, v in weights.items()}
    model.load_state_dict(state_dict, strict=False)

    if not lazy:
        model.eval()

    return model


def sanitize_weights(model_obj, weights, config=None):
    """Helper function to sanitize weights if the model has a sanitize method"""
    if hasattr(model_obj, "sanitize"):
        if config is not None:
            model_obj = model_obj(config)
        weights = model_obj.sanitize(weights)
    return weights


def update_module_configs(model_config, model_class, config, modules):
    """Updates configuration for model modules like text and vision modules.

    Args:
        model_config: The model configuration object that will be updated
        model_class: The model class containing component config classes
        config: Dictionary containing configuration parameters
        modules: List of module names to update configs for (e.g. ["text", "vision"])

    Returns:
        The updated model_config object
    """
    for config_name in modules:
        config_attr = f"{config_name}_config"
        if hasattr(model_config, config_attr):
            config_class = getattr(model_class, f"{config_name.title()}Config")
            setattr(
                model_config, config_attr, config_class.from_dict(config[config_attr])
            )
    return model_config


def get_class_predicate(skip_vision, weights=None):
    if skip_vision:
        return lambda p, m: isinstance(m, nn.Linear) and not (
            "vision_model" in p or "vision_tower" in p
        )
    else:
        if weights:
            return lambda p, m: (
                isinstance(m, nn.Linear)
                and m.weight.shape[-1] % 64 == 0
                and f"{p}.scales" in weights
            )
        else:
            return (
                lambda _, m: isinstance(m, nn.Linear) and m.weight.shape[-1] % 64 == 0
            )


def load(
    path_or_hf_repo: str,
    adapter_path: Optional[str] = None,
    adapter_type: Optional[str] = None,
    lazy: bool = False,
    **kwargs,
) -> Tuple[nn.Module, Union[PreTrainedTokenizer, PreTrainedTokenizerFast]]:
    """
    Load the model and tokenizer from a given path or a huggingface repository.

    Args:
        path_or_hf_repo (Path): The path or the huggingface repository to load the model from.
        tokenizer_config (dict, optional): Configuration parameters specifically for the tokenizer.
            Defaults to an empty dictionary.
        adapter_path (str, optional): Path to the LoRA adapters. If provided, applies LoRA layers
            to the model. Default: ``None``.
        lazy (bool): If False eval the model parameters to make sure they are
            loaded in memory before returning, otherwise they will be loaded
            when needed. Default: ``False``
    Returns:
        Tuple[nn.Module, TokenizerWrapper]: A tuple containing the loaded model and tokenizer.

    Raises:
        FileNotFoundError: If config file or safetensors are not found.
        ValueError: If model class or args class are not found.
    """
    model_path = get_model_path(path_or_hf_repo)

    model = load_model(model_path, lazy, **kwargs)
    if adapter_path is not None:
        # TODO: Support more modules than just language_model
        model = apply_lora_layers(model, adapter_path, adapter_type)
        model.eval()

    image_processor = load_image_processor(model_path, **kwargs)
    processor = load_processor(model_path, True, **kwargs)

    if image_processor is not None:
        processor.image_processor = image_processor

    return model, processor


def load_config(model_path: Union[str, Path], **kwargs) -> dict:
    """Load model configuration from a path or Hugging Face repo.

    Args:
        model_path: Local path or Hugging Face repo ID to load config from
        **kwargs: Additional keyword arguments to pass to the config loader

    Returns:
        dict: Model configuration

    Raises:
        FileNotFoundError: If config.json is not found at the path
    """
    if isinstance(model_path, str):
        model_path = get_model_path(model_path)

    try:
        return AutoConfig.from_pretrained(model_path, **kwargs).to_dict()
    except ValueError:
        try:
            with open(model_path / "config.json", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Config not found at {model_path}") from exc


def load_image_processor(model_path: Union[str, Path], **kwargs) -> BaseImageProcessor:
    if isinstance(model_path, str):
        model_path = get_model_path(model_path)

    if not kwargs:
        config = load_config(model_path, trust_remote_code=True)
    else:
        config = load_config(model_path, **kwargs)

    model_class, _ = get_model_and_args(config)
    image_processor = None

    if hasattr(model_class, "ImageProcessor"):
        import inspect

        init_signature = inspect.signature(model_class.ImageProcessor.__init__)

        if "config" in init_signature.parameters:
            image_processor = model_class.ImageProcessor(config=config)
        else:
            image_processor = model_class.ImageProcessor()

    return image_processor


def load_processor(
    model_path, add_detokenizer=True, **kwargs
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:

    processor = AutoProcessor.from_pretrained(model_path, **kwargs)
    if add_detokenizer:
        detokenizer_class = load_tokenizer(model_path, return_tokenizer=False)
        if "tokenizer" in processor.__dict__.keys():
            processor.detokenizer = detokenizer_class(processor.tokenizer)
        else:
            processor.detokenizer = detokenizer_class(processor)
    return processor


def load_image(image_source: Union[str, Path, BytesIO], timeout: int = 10):
    """
    Helper function to load an image from either a URL or file.
    """
    if isinstance(image_source, str):
        try:
            image_source = BytesIO(base64.b64decode(image_source))
        except Exception as e:
            pass
    if isinstance(image_source, BytesIO) or Path(image_source).is_file():
        # for base64 encoded images
        try:
            image = Image.open(image_source)
        except IOError as e:
            raise ValueError(
                f"Failed to load image from {image_source} with error: {e}"
            ) from e
    elif image_source.startswith(("http://", "https://")):
        try:
            response = requests.get(image_source, stream=True, timeout=timeout)
            response.raise_for_status()
            image = Image.open(response.raw)
        except Exception as e:
            raise ValueError(
                f"Failed to load image from URL: {image_source} with error {e}"
            ) from e
    else:
        raise ValueError(
            f"The image {image_source} must be a valid URL or existing file."
        )

    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def resize_image(img, max_size, regime='ratio'):
    if regime == 'ratio':
        ratio = min(max_size[0] / img.width, max_size[1] / img.height)
        new_size = (int(img.width * ratio), int(img.height * ratio))
    elif regime == 'exact':
        new_size = max_size
    return img.resize(new_size)


def make_shards(weights: dict, max_file_size_gb: int = MAX_FILE_SIZE_GB) -> list:
    """
    Splits the weights into smaller shards.

    Args:
        weights (dict): Model weights.
        max_file_size_gb (int): Maximum size of each shard in gigabytes.

    Returns:
        list: List of weight shards.
    """
    max_file_size_bytes = max_file_size_gb << 30
    shards = []
    shard, shard_size = {}, 0
    for k, v in weights.items():
        if shard_size + v.numel() * v.element_size() > max_file_size_bytes:
            shards.append(shard)
            shard, shard_size = {}, 0
        shard[k] = v
        shard_size += v.numel() * v.element_size()
    shards.append(shard)
    return shards


def save_weights(
    save_path: Union[str, Path],
    weights: Dict[str, Any],
    *,
    donate_weights: bool = False,
) -> None:
    """Save model weights into specified directory."""
    if isinstance(save_path, str):
        save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    shards = make_shards(weights)
    shards_count = len(shards)
    shard_file_format = (
        "model-{:05d}-of-{:05d}.safetensors"
        if shards_count > 1
        else "model.safetensors"
    )

    total_size = sum(v.numel() * v.element_size() for v in weights.values())
    index_data = {"metadata": {"total_size": total_size}, "weight_map": {}}

    # Write the weights and make sure no references are kept other than the
    # necessary ones
    if donate_weights:
        weights.clear()
        del weights

    for i in range(len(shards)):
        shard = shards[i]
        shards[i] = None
        shard_name = shard_file_format.format(i + 1, shards_count)
        shard_path = save_path / shard_name

        safetensors.torch.save_file(shard, str(shard_path), metadata={"format": "pytorch"})

        for weight_name in shard.keys():
            index_data["weight_map"][weight_name] = shard_name
        del shard

    index_data["weight_map"] = {
        k: index_data["weight_map"][k] for k in sorted(index_data["weight_map"])
    }

    with open(save_path / "model.safetensors.index.json", "w") as f:
        json.dump(index_data, f, indent=4)


def upload_to_hub(path: str, upload_repo: str, hf_path: str):
    """
    Uploads the model to Hugging Face hub.

    Args:
        path (str): Local path to the model.
        upload_repo (str): Name of the HF repo to upload to.
        hf_path (str): Path to the original Hugging Face model.
    """
    card = ModelCard.load(hf_path)
    card.data.tags = ["pytorch"] if card.data.tags is None else card.data.tags + ["pytorch"]
    card.text = dedent(
        f"""
        # {upload_repo}
        This model was converted to PyTorch format from [`{hf_path}`]().
        Refer to the [original model card](https://huggingface.co/{hf_path}) for more details on the model.
        """
    )
    card.save(str(Path(path) / "README.md"))

    logging.set_verbosity_info()

    api = HfApi()
    api.create_repo(repo_id=upload_repo, exist_ok=True)
    api.upload_folder(
        folder_path=path,
        repo_id=upload_repo,
        repo_type="model",
    )
    print(f"Upload successful, go to https://huggingface.co/{upload_repo} for details.")


def convert(
    hf_path: str,
    output_path: str = "pytorch_model",
    dtype: str = "float16",
    upload_repo: str = None,
    revision: Optional[str] = None,
    trust_remote_code: bool = True,
):
    """
    Convert a Hugging Face model to PyTorch format.

    Args:
        hf_path (str): Path to the Hugging Face model
        output_path (str): Path to save the converted model
        dtype (str): Data type to convert to
        upload_repo (str): Optional repository to upload the converted model to
        revision (str): Optional revision to use when downloading the model
        trust_remote_code (bool): Whether to trust remote code when loading the model
    """
    print("[INFO] Loading model from Hugging Face")
    model_path = get_model_path(hf_path, revision=revision)
    
    config = load_config(model_path, trust_remote_code=trust_remote_code)
    model_class, _ = get_model_and_args(config)
    
    # Initialize model config
    model_config = model_class.ModelConfig.from_dict(config)
    model = model_class.Model(model_config)
    
    # Load weights
    print("[INFO] Loading weights")
    weights = {}
    for wf in glob.glob(str(model_path / "*.safetensors")):
        with safetensors.safe_open(wf, framework="pt", device="cpu") as f:
            for k in f.keys():
                weights[k] = f.get_tensor(k)

    # Convert dtype
    dtype = getattr(torch, dtype)
    weights = {k: v.to(dtype) for k, v in weights.items()}

    # Save converted model
    print("[INFO] Saving converted model")
    if isinstance(output_path, str):
        output_path = Path(output_path)
    
    save_weights(output_path, weights, donate_weights=True)

    # Copy additional files
    py_files = glob.glob(str(model_path / "*.py"))
    for file in py_files:
        shutil.copy(file, output_path)

    # Save processor
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    processor.save_pretrained(output_path)

    # Save config
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    if upload_repo is not None:
        upload_to_hub(output_path, upload_repo, hf_path)

    print("[INFO] Conversion complete")
