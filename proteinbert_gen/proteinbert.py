import math

import torch
import torch.nn as nn
import numpy as np

from torch import einsum
from einops import rearrange
from einops.layers.torch import Rearrange

from .debugging import print2

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super().__init__()
        self.conv_narrow = nn.Sequential(
            Rearrange('b l d -> b d l'),
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding='same', dilation=1),
            nn.GELU(),
            Rearrange('b d l -> b l d')
        )
        self.conv_wide = nn.Sequential(
            Rearrange('b l d -> b d l'),
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding='same', dilation=5),
            nn.GELU(),
            Rearrange('b d l -> b l d')
        )

    def forward(self, x):
        return self.conv_narrow(x) + self.conv_wide(x)

class GlobalAttention(nn.Module):
    def __init__(self, d_local, d_global, n_heads, d_key):
        super().__init__()
        d_value = d_global // n_heads

        self.to_q = nn.Sequential(
            nn.Linear(d_global, d_key * n_heads, bias=False),
            nn.Tanh()
        )
        self.to_k = nn.Sequential(
            nn.Linear(d_local, d_key * n_heads, bias=False),
            nn.Tanh()
        )
        self.to_v = nn.Sequential(
            nn.Linear(d_local, d_value * n_heads, bias=False),
            nn.GELU()
        )

        self.n_heads = n_heads
        self.d_key = d_key

    def forward(self, x_local, x_global, attention_mask=None):
        q = self.to_q(x_global)
        k = self.to_k(x_local)
        v = self.to_v(x_local)

        q = rearrange(q, 'b (h d) -> b h d', h=self.n_heads)
        k = rearrange(k, 'b l (h d) -> b l h d', h=self.n_heads)
        v = rearrange(v, 'b l (h d) -> b l h d', h=self.n_heads)

        att = einsum('b h d, b l h d -> b h l', q, k) / math.sqrt(self.d_key)

        # attention_mask here
        if attention_mask is not None:
            att_mask = (1.0 - attention_mask) * -10000
            att_mask = att_mask.unsqueeze(1)
            # print("attention and attention mask", att.shape, att_mask.shape)
            att += att_mask

        att = att.softmax(dim=-1)

        x_global = einsum('b h l, b l h d -> b h d', att, v)
        x_global = rearrange(x_global, 'b h d -> b (h d)')
        return x_global


class TransformerLikeBlock(nn.Module):
    def __init__(self, d_local, d_global):
        super().__init__()

        self.wide_and_narrow_conv1d = ConvBlock(d_local, d_local)
        self.dense_and_broadcast = nn.Sequential(
            nn.Linear(d_global, d_local),
            nn.GELU(),
            Rearrange('b d -> b () d')
        )
        self.local_ln1 = nn.LayerNorm(d_local)
        self.local_dense = nn.Sequential(
            Residual(nn.Sequential(nn.Linear(d_local, d_local), nn.GELU())),
            nn.LayerNorm(d_local),
        )

        self.global_dense1 = nn.Sequential(nn.Linear(d_global, d_global), nn.GELU())
        self.global_attention = GlobalAttention(d_local, d_global, n_heads=4, d_key=64)
        self.global_ln1 = nn.LayerNorm(d_global)
        self.global_dense2 = nn.Sequential(
            Residual(nn.Sequential(nn.Linear(d_global, d_global), nn.GELU())),
            nn.LayerNorm(d_global),
        )

    def forward(self, x_local, x_global, attention_mask=None):
        x_local = self.local_ln1(
            x_local + self.wide_and_narrow_conv1d(x_local) + self.dense_and_broadcast(x_global)
        )
        print2("local_ln1", x_local.shape, x_local, tags=["in block"])

        x_local = self.local_dense(x_local)
        print2("local_dense", x_local.shape, x_local, tags=["in block"])

        x_global = self.global_ln1(
            x_global + self.global_dense1(x_global) + self.global_attention(x_local, x_global, attention_mask=attention_mask)
        )
        print2("global_ln1", x_local.shape, x_local, tags=["in block"])

        x_global = self.global_dense2(x_global)
        print2("global_dense2", x_local.shape, x_local, tags=["in block"])

        return x_local, x_global


class ProteinBERT(nn.Module):
    def __init__(
            self,
            vocab_size,
            ann_size,
            d_local=128,
            d_global=512,
        ):
        super().__init__()

        self.vocab_size = vocab_size
        self.ann_size = ann_size

        self.embed_local = nn.Embedding(vocab_size, d_local)
        self.embed_global = nn.Sequential(nn.Linear(ann_size, d_global), nn.GELU())

        self.blocks = nn.ModuleList([TransformerLikeBlock(d_local, d_global) for _ in range(6)])

        self.local_head = nn.Sequential(nn.Linear(d_local, vocab_size))  # NOTE: logits are returned
        self.global_head = nn.Sequential(nn.Linear(d_global, ann_size), nn.Sigmoid())

    def forward(self, x_local, x_global=None, attention_mask=None):
        x_local = self.embed_local(x_local)

        if x_global is None:
            x_global = torch.zeros((x_local.size(0), self.ann_size))
        x_global = self.embed_global(x_global)

        print2("embed_local", x_local.shape, x_local, tags=["inputs"])
        print2("embed_global", x_global.shape, x_global, tags=["inputs"])

        for i, block in enumerate(self.blocks):
            print2(f"=== block {i} ===", tags=["in block"])
            x_local, x_global = block(x_local, x_global, attention_mask=attention_mask)
            print2(f"x_local_{i}", x_local.shape, x_local)

        return self.local_head(x_local)


def load_pretrained_weights(model, pretrained_model_weights):
    # reorganize weights
    for i, weight in enumerate(pretrained_model_weights):
        if i == 2:
            continue
        if len(weight.shape) == 2:
            pretrained_model_weights[i] = np.transpose(weight, (1, 0))
            continue

        if (3 <= i <= 140):
            if i % 23 == 17:
                pretrained_model_weights[i] = np.transpose(weight, (0, 2, 1)).reshape(4 * 64, 512)
                continue
            if i % 23 == 18:
                pretrained_model_weights[i] = np.transpose(weight, (0, 2, 1)).reshape(4 * 64, 128)
                continue
            if i % 23 == 19:
                pretrained_model_weights[i] = np.transpose(weight, (0, 2, 1)).reshape(4 * 128, 128)
                continue

        if len(weight.shape) == 3:
            pretrained_model_weights[i] = np.transpose(weight, (2, 1, 0))

    # convert all to tensors
    for i, weight in enumerate(pretrained_model_weights):
        pretrained_model_weights[i] = torch.from_numpy(weight)

    # load weights
    state = model.state_dict()
    state["embed_local.weight"] = pretrained_model_weights[2]
    state["embed_global.0.weight"] = pretrained_model_weights[0]
    state["embed_global.0.bias"] = pretrained_model_weights[1]

    for block in range(6):
        idx = 3 + block * 23
        state[f"blocks.{block}.wide_and_narrow_conv1d.conv_narrow.1.weight"] = pretrained_model_weights[idx + 2]
        state[f"blocks.{block}.wide_and_narrow_conv1d.conv_narrow.1.bias"] = pretrained_model_weights[idx + 3]
        state[f"blocks.{block}.wide_and_narrow_conv1d.conv_wide.1.weight"] = pretrained_model_weights[idx + 4]
        state[f"blocks.{block}.wide_and_narrow_conv1d.conv_wide.1.bias"] = pretrained_model_weights[idx + 5]
        state[f"blocks.{block}.dense_and_broadcast.0.weight"] = pretrained_model_weights[idx]
        state[f"blocks.{block}.dense_and_broadcast.0.bias"] = pretrained_model_weights[idx + 1]
        state[f"blocks.{block}.local_ln1.weight"] = pretrained_model_weights[idx + 6]
        state[f"blocks.{block}.local_ln1.bias"] = pretrained_model_weights[idx + 7]
        state[f"blocks.{block}.local_dense.0.fn.0.weight"] = pretrained_model_weights[idx + 8]
        state[f"blocks.{block}.local_dense.0.fn.0.bias"] = pretrained_model_weights[idx + 9]
        state[f"blocks.{block}.local_dense.1.weight"] = pretrained_model_weights[idx + 10]
        state[f"blocks.{block}.local_dense.1.bias"] = pretrained_model_weights[idx + 11]
        state[f"blocks.{block}.global_dense1.0.weight"] = pretrained_model_weights[idx + 12]
        state[f"blocks.{block}.global_dense1.0.bias"] = pretrained_model_weights[idx + 13]
        state[f"blocks.{block}.global_attention.to_q.0.weight"] = pretrained_model_weights[idx + 14]
        state[f"blocks.{block}.global_attention.to_k.0.weight"] = pretrained_model_weights[idx + 15]
        state[f"blocks.{block}.global_attention.to_v.0.weight"] = pretrained_model_weights[idx + 16]
        state[f"blocks.{block}.global_ln1.weight"] = pretrained_model_weights[idx + 17]
        state[f"blocks.{block}.global_ln1.bias"] = pretrained_model_weights[idx + 18]
        state[f"blocks.{block}.global_dense2.0.fn.0.weight"] = pretrained_model_weights[idx + 19]
        state[f"blocks.{block}.global_dense2.0.fn.0.bias"] = pretrained_model_weights[idx + 20]
        state[f"blocks.{block}.global_dense2.1.weight"] = pretrained_model_weights[idx + 21]
        state[f"blocks.{block}.global_dense2.1.bias"] = pretrained_model_weights[idx + 22]

    state["local_head.0.weight"] = pretrained_model_weights[141]
    state["local_head.0.bias"] = pretrained_model_weights[142]
    state["global_head.0.weight"] = pretrained_model_weights[143]
    state["global_head.0.bias"] = pretrained_model_weights[144]

    model.load_state_dict(state)

    unfrozen_params = []

    for name, module in model.named_modules():
        print(name, end="")
        # if str(type(module)).find("LayerNorm") != -1:
        #     print(f" ...freezing", end="")
        #     continue

        for param in module.parameters():
            unfrozen_params.append(param)

        print()

    return unfrozen_params
