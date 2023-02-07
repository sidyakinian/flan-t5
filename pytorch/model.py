import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Module, Embedding, Linear, LayerNorm, MultiheadAttention, ModuleList, Softmax
from pdb import set_trace as bkpt
import matplotlib.pyplot as plt
import math

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super(EmbeddingLayer, self).__init__()
        self.word_embedding_layer = Embedding(vocab_size, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.word_embedding_layer(x)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super(MLP, self).__init__()
        self.ff1 = Linear(d_model, d_ff, bias=False)
        self.ff2 = Linear(d_ff, d_model, bias=False)
        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.activation(self.ff1(x))
        return self.ff2(x)


class Norm(nn.Module):
    def __init__(self, hidden_size: int, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: Tensor) -> Tensor:

        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, is_decoder: bool):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = int(d_model / n_heads)
        self.w_q = Linear(d_model, d_model, bias=False)
        self.w_k = Linear(d_model, d_model, bias=False)
        self.w_v = Linear(d_model, d_model, bias=False)
        self.w_o = Linear(d_model, d_model, bias=False)
        self.is_decoder = is_decoder

    def split_heads(self, x: Tensor) -> Tensor:
        batch_size, n, _ = x.size()
        x = x.view((batch_size, n, self.n_heads, self.d_k))
        x = x.transpose(1, 2)
        return x

    def unify_heads(self, x: Tensor) -> Tensor:
        batch_size, _, n, _ = x.size()
        x = x.transpose(1, 2)
        x = x.reshape((batch_size, n, self.d_model))
        return x

    def forward(self, x: Tensor, position_bias: Tensor) -> Tensor:
        _, n, _ = x.size()
        if self.is_decoder:
            Q = self.w_q(x[:, -1, :].unsqueeze(1))
        else:
            Q = self.w_q(x)
        K, V = self.w_k(x), self.w_v(x)
        Q, K, V = self.split_heads(Q), self.split_heads(K), self.split_heads(V)
        scores = (Q @ K.transpose(-1, -2))
        scores += position_bias
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        split_attention = attn_weights @ V
        attention = self.unify_heads(split_attention)
        output = self.w_o(attention)
        return output


class EncDecAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super(EncDecAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = int(d_model / n_heads)
        self.w_q = Linear(d_model, d_model, bias=False)
        self.w_k = Linear(d_model, d_model, bias=False)
        self.w_v = Linear(d_model, d_model, bias=False)
        self.w_o = Linear(d_model, d_model, bias=False)

    def split_heads(self, x: Tensor) -> Tensor:
        batch_size, n, _ = x.size()
        x = x.view((batch_size, n, self.n_heads, self.d_k))
        x = x.transpose(1, 2)
        return x

    def unify_heads(self, x: Tensor) -> Tensor:
        batch_size, _, n, _ = x.size()
        x = x.transpose(1, 2)
        x = x.reshape((batch_size, n, self.d_model))
        return x

    def forward(self, x: Tensor, encoding: Tensor, position_bias: Tensor) -> Tensor:
        _, n, _ = x.size()
        _, encoding_n, _ = encoding.size()
        # Q = self.w_q(x[:, -1, :].unsqueeze(1))
        Q = self.w_q(x) # not sure about cross-attn implementation here
        K, V = self.w_k(encoding), self.w_v(encoding)
        Q, K, V = self.split_heads(Q), self.split_heads(K), self.split_heads(V)
        scores = (Q @ K.transpose(-1, -2))
        scores += position_bias
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        split_attention = attn_weights @ V
        attention = self.unify_heads(split_attention)
        output = self.w_o(attention)
        return output


class EncoderLayer(nn.Module):    
    def __init__(self, d_model: int, d_ff: int, n_heads: int):
        super(EncoderLayer, self).__init__()
        self.self_attention = SelfAttention(d_model=d_model, n_heads=n_heads, is_decoder=False)
        self.norm1 = Norm(d_model)
        self.mlp = MLP(d_model=d_model, d_ff=d_ff)
        self.norm2 = Norm(d_model)

    def forward(self, x: Tensor, position_bias: Tensor) -> Tensor:
        normed_x = self.norm1(x)
        x = x + self.self_attention(normed_x, position_bias=position_bias)
        normed_x = self.norm2(x)
        x = x + self.mlp(normed_x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_heads: int):
        super(DecoderLayer, self).__init__()
        self.self_attention = SelfAttention(d_model=d_model, n_heads=n_heads, is_decoder=True)
        self.norm1 = Norm(d_model)
        self.enc_dec_attention = EncDecAttention(d_model=d_model, n_heads=n_heads)
        self.norm2 = Norm(d_model)
        self.mlp = MLP(d_model=d_model, d_ff=d_ff)
        self.norm3 = Norm(d_model)

    def forward(self, x: Tensor, encoding: Tensor, self_attention_position_bias: Tensor, enc_dec_attention_position_bias: Tensor) -> Tensor:
        # bkpt()
        normed_x = self.norm1(x)
        attn_output = self.self_attention(normed_x, position_bias=self_attention_position_bias)
        x += attn_output
        normed_x = self.norm2(x)
        cross_attn_output = self.enc_dec_attention(normed_x, encoding, position_bias=enc_dec_attention_position_bias)
        x += cross_attn_output
        normed_x = self.norm3(x)
        mlp_output = self.mlp(normed_x)
        x += mlp_output
        return x


class Encoder(nn.Module):
    def __init__(self, n_layers: int, d_model: int, d_ff: int, n_heads: int):
        super(Encoder, self).__init__()
        self.layers = ModuleList([EncoderLayer(d_model=d_model, d_ff=d_ff, n_heads=n_heads) for i in range(n_layers)])
        self.self_attention_position_bias = None
        self.self_attention_relative_attention_embedding = Embedding(32, n_heads)

    def forward(self, x: Tensor) -> Tensor:
        _, n, _ = x.size()
        self.self_attention_position_bias = self.compute_bias(n, n)
        for layer in self.layers:
            x = layer(x, self.self_attention_position_bias)
        return x

    def relative_position_bucket(self, relative_position: Tensor, bidirectional=True, num_buckets=32, max_distance=128) -> Tensor:
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))

        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length: int, key_length: int, device=None) -> Tensor:
        if device is None:
            device = self.self_attention_relative_attention_embedding.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self.relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=True,
        )
        values = self.self_attention_relative_attention_embedding(relative_position_bucket)  # shape (query_length, key_length, n_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, n_heads, query_length, key_length)
        return values


class Decoder(nn.Module):
    def __init__(self, n_layers: int, d_model: int, d_ff: int, n_heads: int):
        super(Decoder, self).__init__()
        self.layers = ModuleList([DecoderLayer(d_model=d_model, d_ff=d_ff, n_heads=n_heads) for i in range(n_layers)])
        self.self_attention_position_bias = None
        self.self_attention_relative_attention_embedding = Embedding(32, n_heads)
        self.enc_dec_attention_position_bias = None
        self.enc_dec_attention_relative_attention_embedding = Embedding(32, n_heads)

    def forward(self, x: Tensor, encoding: Tensor) -> Tensor:
        _, n, _ = x.size()
        _, encoding_n, _ = encoding.size()
        self.self_attention_position_bias = self.compute_bias(1, n, enc_dec=False)
        self.enc_dec_attention_position_bias = self.compute_bias(1, encoding_n, enc_dec=True)
        for layer in self.layers:
            x = layer(x, 
                      encoding, 
                      self_attention_position_bias=self.self_attention_position_bias, 
                      enc_dec_attention_position_bias=self.enc_dec_attention_position_bias)
        return x[:, -1, :].unsqueeze(1)
    
    def relative_position_bucket(self, relative_position: Tensor, bidirectional=True, num_buckets=32, max_distance=128) -> Tensor:
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))

        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length: int, key_length: int, device=None, enc_dec=True) -> Tensor:
        if device is None:
            if enc_dec:
                device = self.enc_dec_attention_relative_attention_embedding.weight.device
            else:
                device = self.self_attention_relative_attention_embedding.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self.relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=False,
        )
        if enc_dec:
            values = self.enc_dec_attention_relative_attention_embedding(relative_position_bucket)  # shape (query_length, key_length, n_heads)
        else:
            values = self.self_attention_relative_attention_embedding(relative_position_bucket)  # shape (query_length, key_length, n_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, n_heads, query_length, key_length)
        return values


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, n_layers: int, d_model: int, d_ff: int, n_heads: int):
        super(Transformer, self).__init__()
        self.embedding_layer = EmbeddingLayer(vocab_size=vocab_size, d_model=d_model)
        self.encoder = Encoder(n_layers=n_layers, d_model=d_model, d_ff=d_ff, n_heads=n_heads)
        self.final_encoder_layer_norm = Norm(d_model)
        self.decoder = Decoder(n_layers=n_layers, d_model=d_model, d_ff=d_ff, n_heads=n_heads)
        self.final_decoder_layer_norm = Norm(d_model)
        self.d_model = d_model
        self.lm_head = Linear(d_model, vocab_size, bias=False)

    def forward(self, x: Tensor, max_tokens=10, debug=False) -> Tensor:
        x_emb = self.embedding_layer(x)
        encoding = self.encoder(x_emb)
        encoding = self.final_encoder_layer_norm(encoding)
        outputs = torch.IntTensor([[0]]).to("cuda")
        # for _ in range(max_tokens):
        while outputs[:, -1] != 1:
            if debug:
                bkpt()
            decoder_inputs_embedding = self.embedding_layer(outputs)
            decoding = self.decoder(decoder_inputs_embedding, encoding)
            decoding = self.final_decoder_layer_norm(decoding)
            decoding *= (self.d_model ** -0.5)
            next_logits = self.lm_head(decoding)
            next_token = torch.argmax(next_logits, dim=-1)
            outputs = torch.cat([outputs, next_token], dim=-1)
        return outputs
