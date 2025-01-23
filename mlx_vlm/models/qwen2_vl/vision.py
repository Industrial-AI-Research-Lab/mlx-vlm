import inspect
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


@dataclass
class VisionConfig:
    model_type: str = "qwen2_vl"
    depth: int = 32
    embed_dim: int = 1280
    hidden_size: int = 1536
    num_heads: int = 16
    image_size: int = 384
    patch_size: int = 14
    vocab_size: int = 32000
    mlp_ratio: float = 4.0
    in_channels: int = 3
    layer_norm_eps: float = 1e-6
    spatial_patch_size: int = 14
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2

    @classmethod
    def from_dict(cls, params):
        # Ensure all fields have values, using defaults if not provided
        config_dict = {
            field: params.get(field, getattr(cls, field))
            for field in cls.__dataclass_fields__
        }
        return cls(**config_dict)


def check_array_shape(arr):
    shape = arr.shape

    # Check if the shape has 4 dimensions
    if len(shape) not in [4, 5]:
        return False

    B, out_channels, kH, KW, t = shape

    if t == 3:
        return True

    # Check if out_channels is the largest, and kH and KW are the same
    if (out_channels >= kH) and (out_channels >= KW) and (kH == KW):
        return True
    else:
        return False


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb_vision(tensor, freqs) -> torch.Tensor:
    orig_dtype = tensor.dtype

    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    cos = cos.unsqueeze(1)
    cos = cos.repeat(1, 1, 2)
    cos = cos.unsqueeze(0)

    sin = sin.unsqueeze(1)
    sin = sin.repeat(1, 1, 2)
    sin = sin.unsqueeze(0)

    output = (tensor * cos) + (rotate_half(tensor) * sin)
    return output.to(orig_dtype)


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta

    def __call__(self, seqlen: int) -> torch.Tensor:
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        seq = torch.arange(int(seqlen), dtype=inv_freq.dtype)
        freqs = torch.outer(seq, inv_freq)
        return freqs


class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = torch.nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
        )

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.reshape(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        ).permute(0, 1, 2, 3, 4)  # [B, C, T, H, W]

        hidden_states = self.proj(hidden_states)
        hidden_states = hidden_states.reshape(-1, self.embed_dim)
        return hidden_states


class PatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = torch.nn.LayerNorm(context_dim, eps=1e-6)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_size, dim),
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_q(x).reshape(-1, self.hidden_size)
        x = self.mlp(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def __call__(
        self, x: torch.Tensor, cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor = None
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, num_heads, N, head_dim

        if rotary_pos_emb is not None:
            q = apply_rotary_pos_emb_vision(q, rotary_pos_emb)
            k = apply_rotary_pos_emb_vision(k, rotary_pos_emb)

        # Scaled dot product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Create attention mask based on cu_seqlens
        attention_mask = torch.ones((B, 1, N, N), dtype=x.dtype, device=x.device)
        for i in range(1, len(cu_seqlens)):
            start_idx = cu_seqlens[i-1]
            end_idx = cu_seqlens[i]
            attention_mask[:, :, start_idx:end_idx, end_idx:] = 0
            attention_mask[:, :, end_idx:, start_idx:end_idx] = 0

        attn = attn.masked_fill(attention_mask == 0, float('-inf'))
        attn = torch.softmax(attn, dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.activation_fn = nn.GELU()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation_fn(self.fc1(x))
        x = self.fc2(x)
        return x


class Qwen2VLVisionBlock(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.embed_dim, eps=1e-6)
        mlp_hidden_dim = int(config.embed_dim * config.mlp_ratio)

        self.attn = Attention(dim=config.embed_dim, num_heads=config.num_heads)
        self.mlp = MLP(dim=config.embed_dim, hidden_dim=mlp_hidden_dim)

    def __call__(self, hidden_states, cu_seqlens, rotary_pos_emb) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()

        self.spatial_merge_size = config.spatial_merge_size
        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )

        self.rotary_pos_emb = VisionRotaryEmbedding(config.embed_dim // config.num_heads)

        mlp_hidden_dim = int(config.embed_dim * config.mlp_ratio)
        self.blocks = nn.ModuleList([
            Qwen2VLVisionBlock(config) for _ in range(config.depth)
        ])

        self.merger = PatchMerger(
            dim=config.hidden_size,
            context_dim=config.embed_dim,
            spatial_merge_size=config.spatial_merge_size,
        )

    def rot_pos_emb(self, grid_thw):
        pos_ids = []

        for t, h, w in grid_thw:
            h, w = int(h), int(w)  # Ensure h and w are integers
            hpos_ids = torch.arange(h, dtype=torch.long, device=grid_thw.device).unsqueeze(1)
            hpos_ids = hpos_ids.repeat(1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w, dtype=torch.long, device=grid_thw.device).unsqueeze(0)
            wpos_ids = wpos_ids.repeat(h, 1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()

            stacked_pos_ids = torch.stack([hpos_ids, wpos_ids], dim=-1)
            pos_ids.append(stacked_pos_ids.repeat(int(t), 1))

        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = torch.max(grid_thw[:, 1:])
        rotary_pos_emb_full = self.rotary_pos_emb(int(max_grid_size))

        rotary_pos_emb_full = rotary_pos_emb_full[pos_ids.long()]
        return rotary_pos_emb_full.reshape(pos_ids.shape[0], -1)

    def __call__(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
    ) -> torch.Tensor:

        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        # Assuming grid_thw has shape (batch_size, 3)
        batch_size = grid_thw.shape[0]

        # Calculate cu_seqlens for each item in the batch
        cu_seqlens = []
        for i in range(batch_size):
            seq_len = int(grid_thw[i, 1] * grid_thw[i, 2])
            cu_seqlens.extend([seq_len] * int(grid_thw[i, 0]))

        # Convert to tensor and compute cumulative sums
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=hidden_states.device)
        cu_seqlens = torch.cumsum(cu_seqlens, dim=0)
        cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.int32, device=hidden_states.device), cu_seqlens])

        encoder_states = (hidden_states,) if output_hidden_states else None

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb
            )
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

        return self.merger(hidden_states)

    @staticmethod
    def sanitize(weights):
        sanitized_weights = {}
        for k, v in weights.items():
            # Skip non-vision model weights
            if "vision_model" not in k and "vision_tower" not in k:
                sanitized_weights[k] = v
                continue

            # Convert vision_model to vision_tower in the key
            new_k = k.replace("vision_model.", "vision_tower.")
            
            # Handle special cases for tensor transformations
            if "patch_embeddings.projection.weight" in k:
                # Convert Conv3d weights from MLX to PyTorch format
                sanitized_weights[new_k] = v.permute(0, 4, 1, 2, 3)
            elif "position_embeddings" in k and len(v.shape) == 3:
                sanitized_weights[new_k] = v.permute(1, 0, 2)
            elif "self_attn.qkv_proj.weight" in k:
                # Split QKV into separate Q, K, V
                qkv = v.reshape(3, -1, v.shape[-1])
                q, k, v_weight = qkv[0], qkv[1], qkv[2]
                base_k = new_k.replace("qkv_proj", "")
                sanitized_weights[base_k + "q_proj.weight"] = q
                sanitized_weights[base_k + "k_proj.weight"] = k
                sanitized_weights[base_k + "v_proj.weight"] = v_weight
            elif "self_attn.qkv_proj.bias" in k:
                # Split QKV bias into separate Q, K, V
                qkv = v.reshape(3, -1)
                q, k, v_weight = qkv[0], qkv[1], qkv[2]
                base_k = new_k.replace("qkv_proj", "")
                sanitized_weights[base_k + "q_proj.bias"] = q
                sanitized_weights[base_k + "k_proj.bias"] = k
                sanitized_weights[base_k + "v_proj.bias"] = v_weight
            else:
                sanitized_weights[new_k] = v
        return sanitized_weights
