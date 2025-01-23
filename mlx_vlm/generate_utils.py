import time
from typing import Dict, Generator, List, Optional, Union

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer

from .utils import load_image, resize_image


def prepare_inputs(
    processor,
    images,
    prompts,
    image_token_index,
    resize_shape=None,
    regime='ratio'
):
    """Prepare inputs for the model."""
    if not isinstance(images, list):
        images = [images]

    # Process images
    image_processor = (
        processor.image_processor if hasattr(processor, "image_processor") else None
    )
    processed_images = []
    for img in images:
        img = load_image(img)
        if resize_shape is not None:
            img = resize_image(img, resize_shape, regime=regime)
        if image_processor is not None:
            img = image_processor.preprocess(images=[img])[0]
        processed_images.append(img)

    # Convert to tensors
    pixel_values = torch.stack([torch.tensor(img) for img in processed_images])

    # Process text
    if not isinstance(prompts, list):
        prompts = [prompts]

    # Prepare inputs based on processor type
    if hasattr(processor, "image_processor"):
        processor.pad_token = processor.eos_token
        text_chunks = [
            [processor(chunk).input_ids for chunk in prompt.split("<image>")]
            for prompt in prompts
        ]

        # Find the maximum length for padding
        max_length = max(
            sum(len(chunk) for chunk in chunks) + 1 for chunks in text_chunks
        )

        # Pad and create input_ids
        input_ids = []
        for chunks in text_chunks:
            ids = chunks[0] + [image_token_index] + chunks[1]
            padding = [processor.pad_token_id] * (max_length - len(ids))
            input_ids.append(torch.tensor(ids + padding))

        input_ids = torch.stack(input_ids)
        attention_mask = (input_ids != processor.pad_token_id).int()

    else:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        inputs = processor(
            text=prompts,
            images=processed_images,
            padding=True,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        if "images" in inputs:
            pixel_values = inputs["images"]

    return {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "attention_mask": attention_mask,
    }


def generate(
    model: nn.Module,
    processor: PreTrainedTokenizer,
    prompt: str,
    image: Union[str, List[str]] = None,
    max_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
    **kwargs,
) -> str:
    """Generate text from the model."""
    device = next(model.parameters()).device
    
    # Prepare inputs
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    image_token_index = getattr(model.config, "image_token_index", None)
    
    if not image:
        input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
        pixel_values = attention_mask = None
    else:
        inputs = prepare_inputs(
            processor, image, prompt, image_token_index,
            resize_shape=kwargs.get("resize_shape")
        )
        input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values"]
        attention_mask = inputs["attention_mask"]

    # Move inputs to device
    input_ids = input_ids.to(device)
    if pixel_values is not None:
        pixel_values = pixel_values.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # Prepare generation config
    generation_config = {
        "max_new_tokens": max_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    if temperature > 0:
        generation_config.update({
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
        })

    # Filter out unsupported kwargs
    supported_kwargs = {
        "max_kv_size", "vision_merge_ratio", "vision_filter_ratio"
    }
    model_kwargs = {k: v for k, v in kwargs.items() if k in supported_kwargs}

    # Generate
    with torch.no_grad():
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values
            
        outputs = model.generate(
            **model_inputs,
            **generation_config,
            **model_kwargs
        )

    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if prompt in generated_text:
        generated_text = generated_text[len(prompt):].strip()
    
    return generated_text


def stream_generate(
    model: nn.Module,
    processor: PreTrainedTokenizer,
    prompt: str,
    image: Union[str, List[str]] = None,
    **kwargs,
) -> Generator[str, None, None]:
    """Stream text generation from the model."""
    text = ""
    for token in generate(model, processor, prompt, image, **kwargs):
        text += token
        yield text 