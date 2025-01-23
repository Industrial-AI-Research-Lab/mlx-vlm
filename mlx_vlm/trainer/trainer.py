import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Dict
from PIL import Image

import torch
import torch.nn as nn
from safetensors.torch import save_file
from torch.utils.data import Dataset as TorchDataset
from transformers import PreTrainedTokenizer

from ..models.base import BaseImageProcessor


def process_image(img, resize_shape, image_processor):
    """Process image for the model."""
    if isinstance(img, str):
        img = Image.open(img).convert('RGB')
    if resize_shape is not None and not isinstance(image_processor, BaseImageProcessor):
        ratio = min(resize_shape[0] / img.width, resize_shape[1] / img.height)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size)
    return img


def get_prompt(model_type, processor, conversation):
    if model_type == "paligemma":
        return conversation

    if "chat_template" in processor.__dict__.keys():
        prompt = processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )
    elif "tokenizer" in processor.__dict__.keys():
        prompt = processor.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )

    return prompt


class Dataset(TorchDataset):
    def __init__(
        self,
        dataset,
        config: dict,
        processor: PreTrainedTokenizer,
        image_processor=None,
        image_resize_shape=None,
    ):
        self.dataset = dataset
        self.config = config
        self.processor = processor
        self.image_processor = image_processor
        self.image_resize_shape = image_resize_shape

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Process images
        images = item["images"]
        if not isinstance(images, list):
            images = [images]
        
        processed_images = []
        for image in images:
            img = process_image(image, self.image_resize_shape, self.image_processor)
            if hasattr(self.processor, "image_processor"):
                img = self.processor.image_processor.preprocess(images=[img])[0]
            processed_images.append(img)
        
        pixel_values = torch.stack([torch.tensor(img) for img in processed_images])

        # Process text
        if isinstance(item["messages"], str):
            messages = json.loads(item["messages"])
        else:
            messages = item["messages"]

        model_inputs = self.processor(
            text=messages,
            padding=True,
            return_tensors="pt",
        )

        # Add pixel values to model inputs
        model_inputs["pixel_values"] = pixel_values

        # Convert all tensors to the same device
        model_inputs = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v 
                       for k, v in model_inputs.items()}

        return model_inputs


def grad_checkpoint(layer):
    """
    Update all instances of type(layer) to use gradient checkpointing.
    """
    fn = type(layer).__call__

    def checkpointed_fn(model, *args, **kwargs):
        def inner_fn(params, *args, **kwargs):
            model.update(params)
            return fn(model, *args, **kwargs)

        return mx.checkpoint(inner_fn)(model.trainable_parameters(), *args, **kwargs)

    type(layer).__call__ = checkpointed_fn


@dataclass
class TrainingArgs:
    batch_size: int = field(default=4, metadata={"help": "Minibatch size."})
    iters: int = field(default=100, metadata={"help": "Iterations to train for."})
    val_batches: int = field(
        default=25,
        metadata={
            "help": "Number of validation batches, -1 uses the entire validation set."
        },
    )
    steps_per_report: int = field(
        default=10,
        metadata={"help": "Number of training steps between loss reporting."},
    )
    steps_per_eval: int = field(
        default=200, metadata={"help": "Number of training steps between validations."}
    )
    steps_per_save: int = field(
        default=100, metadata={"help": "Save the model every number steps"}
    )
    max_seq_length: int = field(
        default=2048, metadata={"help": "Maximum sequence length."}
    )
    adapter_file: str = field(
        default="adapters.safetensors",
        metadata={"help": "Save/load path for the trained adapter weights."},
    )
    grad_checkpoint: bool = field(
        default=False,
        metadata={"help": "Use gradient checkpointing to reduce memory use."},
    )


def default_loss(model, inputs, targets, lengths):
    logits = model(inputs)
    logits = logits.astype(mx.float32)

    length_mask = mx.arange(inputs.shape[1])[None, :] < lengths[:, None]

    ce = nn.losses.cross_entropy(logits, targets) * length_mask
    ntoks = length_mask.sum()
    ce = ce.sum() / ntoks

    return ce, ntoks


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device

    def train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        self.optimizer.zero_grad()
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}

        # Forward pass
        outputs = self.model(**batch)
        loss = outputs.loss

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return loss

    def save_adapter(self, save_path: Union[str, Path]):
        """Save the LoRA adapter weights."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save adapter config
        config = {
            "rank": self.model.config.lora["rank"],
            "alpha": self.model.config.lora["alpha"],
            "dropout": self.model.config.lora["dropout"],
        }
        
        with open(save_path / "adapter_config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Save adapter weights
        adapter_weights = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                adapter_weights[name] = param.detach().cpu()

        save_file(adapter_weights, save_path / "adapters.safetensors")
