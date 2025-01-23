import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, Union, List, Tuple
import json
from pathlib import Path
import glob
from huggingface_hub import snapshot_download
from transformers import AutoConfig
import safetensors

from .vision import VisionModel, VisionConfig
from .language import LanguageModel, TextConfig

@dataclass
class ModelConfig:
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str
    ignore_index: int = -100
    image_token_index: int = 151857
    vocab_size: int = 151936

    @classmethod
    def from_dict(cls, params):
        return cls(**{k: v for k, v in params.items() if k in cls.__dataclass_fields__})

class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.vision_model = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        
        # Initialize projector from vision to language space
        self.mm_projector = nn.Sequential(
            nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size),
            nn.GELU(),
            nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size)
        )
        
    def get_input_embeddings(self, input_ids=None, pixel_values=None):
        if pixel_values is None:
            return self.language_model.embed_tokens(input_ids)
            
        inputs_embeds = self.language_model.embed_tokens(input_ids)
        
        # Process image through vision model
        vision_outputs = self.vision_model(pixel_values)
        image_features = vision_outputs[1]  # Get hidden states
        
        # Project image features to language space
        image_features = self.mm_projector(image_features)
        
        # Prepare inputs by merging text and image embeddings
        merged_embeds = self._prepare_inputs_for_multimodal(
            image_features, inputs_embeds, input_ids
        )
        return merged_embeds
        
    def _prepare_inputs_for_multimodal(self, image_features, inputs_embeds, input_ids):
        image_token_index = self.config.image_token_index
        batch_size, seq_length, embed_dim = inputs_embeds.shape
        num_images, num_image_patches, _ = image_features.shape
        
        # Find positions of <image> tokens
        image_positions = torch.argmax((input_ids == image_token_index).float(), dim=1)
        
        final_embeddings = []
        for b in range(batch_size):
            text_segments = []
            position = int(image_positions[b].item())
            
            # Add text before image
            text_segments.append(inputs_embeds[b:b+1, :position])
            # Add image features
            text_segments.append(image_features[b:b+1])
            # Add text after image
            text_segments.append(inputs_embeds[b:b+1, position+1:])
            
            batch_embeddings = torch.cat(text_segments, dim=1)
            final_embeddings.append(batch_embeddings)
            
        return torch.cat(final_embeddings, dim=0)
        
    def forward(self, input_ids, pixel_values, attention_mask=None):
        inputs_embeds = self.get_input_embeddings(input_ids, pixel_values)
        outputs = self.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds
        )
        return outputs
        
    @staticmethod
    def from_pretrained(path_or_hf_repo: str):
        path = Path(path_or_hf_repo)
        if not path.exists():
            path = Path(
                snapshot_download(
                    repo_id=path_or_hf_repo,
                    allow_patterns=[
                        "*.json",
                        "*.safetensors",
                        "*.py",
                        "tokenizer.model",
                        "*.tiktoken",
                    ],
                )
            )
            
        # Load config
        with open(path / "config.json", "r") as f:
            config = json.load(f)
            
        # Load vision and text configs
        vision_config = AutoConfig.from_pretrained(path, subfolder="vision_config")
        text_config = AutoConfig.from_pretrained(path, subfolder="text_config")
        
        # Convert configs to dicts and ensure required fields
        vision_config_dict = vision_config.to_dict()
        text_config_dict = text_config.to_dict()
        
        # Ensure required fields for text config
        required_text_fields = {
            "model_type": text_config_dict.get("model_type", "qwen2_vl"),
            "hidden_size": text_config_dict.get("hidden_size", 2048),
            "num_hidden_layers": text_config_dict.get("num_hidden_layers", 24),
            "intermediate_size": text_config_dict.get("intermediate_size", 8192),
            "num_attention_heads": text_config_dict.get("num_attention_heads", 16),
            "rms_norm_eps": text_config_dict.get("rms_norm_eps", 1e-6),
            "vocab_size": text_config_dict.get("vocab_size", 151936),
            "num_key_value_heads": text_config_dict.get("num_key_value_heads", None),
            "rope_theta": text_config_dict.get("rope_theta", 10000),
            "rope_traditional": text_config_dict.get("rope_traditional", False),
            "rope_scaling": text_config_dict.get("rope_scaling", None),
            "tie_word_embeddings": text_config_dict.get("tie_word_embeddings", False)
        }
        
        # Update main config
        config["vision_config"] = vision_config_dict
        config["text_config"] = required_text_fields
        
        # Create model config
        model_config = ModelConfig.from_dict(config)
        model_config.vision_config = VisionConfig.from_dict(config["vision_config"])
        model_config.text_config = TextConfig.from_dict(config["text_config"])
        
        # Initialize model
        model = Model(model_config)
        
        # Load weights
        weight_files = glob.glob(str(path / "*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(f"No safetensors found in {path}")
            
        weights = {}
        for wf in weight_files:
            with safetensors.safe_open(wf, framework="pt", device="cpu") as f:
                for k in f.keys():
                    weights[k] = f.get_tensor(k)
                    
        # Convert weights to PyTorch format
        state_dict = {}
        for k, v in weights.items():
            if isinstance(v, (list, tuple)):
                v = torch.tensor(v)
            if "patch_embedding.weight" in k or "patch_embed.weight" in k:
                # Convert from [out_channels, kH, kW, in_channels] to [out_channels, in_channels, kH, kW]
                v = v.permute(0, 3, 1, 2)
            state_dict[k] = v
            
        # Load state dict
        model.load_state_dict(state_dict, strict=False)
        return model 