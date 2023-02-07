import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, List, Optional, Sequence
from jax import random
from typing import Sequence, List, Dict, Callable
import numpy as np
from jaxtyping import Array, Float, PyTree
from jax import Array
from dataclasses import dataclass

class EmbeddingLayer(nn.Module):
    config: TransformerConfig

    def setup(self):
        self.embedding = nn.Embed(num_embeddings=self.config.vocab_size, features=self.config.d_model)

    def __call__(self, x):
        return self.embedding(x)


class PositionEmbeddingLayer(nn.Module):
    config: TransformerConfig

    def setup(self):
        self.pos_embedding = nn.Embed(num_embeddings=self.config.num_relative_pos, features=self.config.n_heads)

    def __call__(self, x):
        return self.pos_embedding(x)


class MLP(nn.Module):
    config: TransformerConfig

    def setup(self):
        self.ff1 = nn.Dense(self.config.d_ff, use_bias=False)
        self.ff2 = nn.Dense(self.config.d_model, use_bias=False)

    def __call__(self, x):
        x = nn.relu(self.ff1(x))
        return self.ff2(x)

class LayerNorm(nn.Module):
    config: TransformerConfig
    weight_init: Callable[..., jnp.ndarray] = jax.nn.initializers.ones

    def setup(self):
        self.weight = self.param("weight", self.weight_init, (self.config.d_model,))

    def __call__(self, x: Array) -> Array:
        variance = jnp.mean(jax.lax.square(x), axis=-1, keepdims=True)
        norm_x = x * jax.lax.rsqrt(variance + self.config.eps)
        return self.weight * norm_x


class SelfAttention(nn.Module):
    config: TransformerConfig
    is_decoder: bool

    def setup(self):
        self.w_q = nn.Dense(self.config.d_model, use_bias=False)
        self.w_k = nn.Dense(self.config.d_model, use_bias=False)
        self.w_v = nn.Dense(self.config.d_model, use_bias=False)
        self.w_o = nn.Dense(self.config.d_model, use_bias=False)

    def __call__(self, x: Array, position_bias: Array) -> Array:
        batch_size, n, _ = x.shape
        if self.is_decoder:
            x = jax.lax.expand_dims(x[:, -1, :], 1)
            Q = self.w_q(x)
        else:
            Q = self.w_q(x)
        K, V = self.w_k(x), self.w_v(x)
        Q, K, V = self.split_heads(Q), self.split_heads(K), self.split_heads(V)
        # Q is of shape (batch_size, n_heads, 1 or n, d_k)
        # K, V are of shape (batch_size, n_heads, n, d_k)
        scores = jax.lax.dot_general(Q, K, (((3,), (3,)), ((0,1), (0,1))))
        scores += position_bias
        attn_weights = nn.softmax(scores, axis=-1)
        split_attn = jax.lax.dot_general(attn_weights, V, (((3,), (2,)), ((0,1), (0,1))))
        attention = self.unify_heads(split_attn)
        output = self.w_o(attention)
        return output

    def split_heads(self, x: Array) -> Array:
        batch_size, n, _ = x.shape
        x = jax.lax.reshape(x, (batch_size, n, self.config.n_heads, self.config.d_k))
        x = jax.lax.transpose(x, (0, 2, 1, 3))
        return x

    def unify_heads(self, x: Array) -> Array:
        batch_size, n_heads, n, _ = x.shape
        x = jax.lax.transpose(x, (0, 2, 1, 3))
        x = jax.lax.reshape(x, (batch_size, n, self.config.d_model))
        return x


class EncoderLayer(nn.Module):
    config: TransformerConfig

    def setup(self):
        self.self_attn = SelfAttention(config=self.config, is_decoder=False)
        self.mlp = MLP(config=self.config)
        self.norm1 = LayerNorm(config=self.config)
        self.norm2 = LayerNorm(config=self.config)

    def __call__(self, x: Array, position_bias: Array) -> Array:
        normed_x = self.norm1(x)
        x += self.self_attn(normed_x, position_bias=position_bias)
        normed_x = self.norm2(x)
        x += self.mlp(normed_x)
        return x


class Encoder(nn.Module):
    config: TransformerConfig

    def setup(self):
        self.encoder_layers = [EncoderLayer(config=self.config) for _ in range(self.config.num_layers)]
        self.self_attention_relative_attention_embedding = PositionEmbeddingLayer(config=self.config)

    def __call__(self, x: Array) -> Array:
        _, n, _ = x.shape
        self_attention_position_bias = self.compute_bias(n, n)
        for i in range(self.config.num_layers):
            x = self.encoder_layers[i](x, self_attention_position_bias)
        return x

    def _relative_position_bucket(self, relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0) * num_buckets
            relative_position = jnp.abs(relative_position)
        else:
            relative_position = -jnp.clip(relative_position, a_max=0)

        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        relative_position_if_large = max_exact + (
            jnp.log(relative_position / max_exact) / jnp.log(max_distance / max_exact) * (num_buckets - max_exact)
        )
        relative_position_if_large = jnp.clip(relative_position_if_large, a_max=num_buckets - 1)

        relative_buckets += jnp.where(is_small, relative_position, relative_position_if_large)

        return relative_buckets.astype("i4")

    def compute_bias(self, query_length, key_length):
        context_position = jnp.arange(query_length, dtype="i4")[:, None]
        memory_position = jnp.arange(key_length, dtype="i4")[None, :]

        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(
            relative_position=relative_position,
            bidirectional=True,
            num_buckets=self.config.num_relative_pos,
            max_distance=128
        )

        values = self.self_attention_relative_attention_embedding(relative_position_bucket)
        values = values.transpose((2, 0, 1))[None, :, :, :]
        return values


class CrossAttention(nn.Module):
    config: TransformerConfig

    def setup(self):
        self.w_q = nn.Dense(self.config.d_model, use_bias=False)
        self.w_k = nn.Dense(self.config.d_model, use_bias=False)
        self.w_v = nn.Dense(self.config.d_model, use_bias=False)
        self.w_o = nn.Dense(self.config.d_model, use_bias=False)

    def __call__(self, x: Array, encoding: Array, position_bias: Array) -> Array:
        batch_size, n, _ = x.shape
        Q = self.w_q(x)
        K, V = self.w_k(encoding), self.w_v(encoding)
        Q, K, V = self.split_heads(Q), self.split_heads(K), self.split_heads(V)
        # Q is of shape (batch_size, n_heads, 1 or n, d_k)
        # K, V are of shape (batch_size, n_heads, n, d_k)
        scores = jax.lax.dot_general(Q, K, (((3,), (3,)), ((0,1), (0,1))))
        scores += position_bias
        attn_weights = nn.softmax(scores, axis=-1)
        split_attn = jax.lax.dot_general(attn_weights, V, (((3,), (2,)), ((0,1), (0,1))))
        attention = self.unify_heads(split_attn)
        output = self.w_o(attention)
        return output

    def split_heads(self, x: Array) -> Array:
        batch_size, n, _ = x.shape
        x = jax.lax.reshape(x, (batch_size, n, self.config.n_heads, self.config.d_k))
        x = jax.lax.transpose(x, (0, 2, 1, 3))
        return x

    def unify_heads(self, x: Array) -> Array:
        batch_size, _, n, _ = x.shape
        x = jax.lax.transpose(x, (0, 2, 1, 3))
        x = jax.lax.reshape(x, (batch_size, n, self.config.d_model))
        return x


class DecoderLayer(nn.Module):
    config: TransformerConfig

    def setup(self):
        self.self_attn = SelfAttention(config=self.config, is_decoder=False)
        self.cross_attn = CrossAttention(config=self.config)
        self.mlp = MLP(config=self.config)
        self.norm1 = LayerNorm(config=self.config)
        self.norm2 = LayerNorm(config=self.config)
        self.norm3 = LayerNorm(config=self.config)

    def __call__(self, x: Array, encoding: Array, self_attention_position_bias: Array, cross_attention_position_bias: Array) -> Array:
        normed_x = self.norm1(x)
        x += self.self_attn(normed_x, position_bias=self_attention_position_bias)
        normed_x = self.norm2(x)
        x += self.cross_attn(normed_x, encoding, position_bias=cross_attention_position_bias)
        normed_x = self.norm3(x)
        x += self.mlp(normed_x)
        return x


class Decoder(nn.Module):
    config: TransformerConfig

    def setup(self):
        self.decoder_layers = [DecoderLayer(config=self.config) for _ in range(self.config.num_layers)]
        self.self_attention_relative_attention_embedding = PositionEmbeddingLayer(config=self.config)
        self.enc_dec_attention_relative_attention_embedding = PositionEmbeddingLayer(config=self.config)

    def __call__(self, x: Array, encoding: Array) -> Array:
        _, n, _ = x.shape
        _, encoding_n, _ = encoding.shape
        self_attention_position_bias = self.compute_bias(1, n, cross=False)
        enc_dec_attention_position_bias = self.compute_bias(1, encoding_n, cross=True)
        for i in range(self.config.num_layers):
            x = self.decoder_layers[i](x,
                                       encoding,
                                       self_attention_position_bias,
                                       enc_dec_attention_position_bias)
        return jax.lax.expand_dims(x[:, -1, :], [1])

    def _relative_position_bucket(self, relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0) * num_buckets
            relative_position = jnp.abs(relative_position)
        else:
            relative_position = -jnp.clip(relative_position, a_max=0)
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            jnp.log(relative_position / max_exact) / jnp.log(max_distance / max_exact) * (num_buckets - max_exact)
        )
        relative_position_if_large = jnp.clip(relative_position_if_large, a_max=num_buckets - 1)

        relative_buckets += jnp.where(is_small, relative_position, relative_position_if_large)

        return relative_buckets.astype("i4")

    def compute_bias(self, query_length, key_length, cross: bool):
        context_position = jnp.arange(query_length, dtype="i4")[:, None]
        memory_position = jnp.arange(key_length, dtype="i4")[None, :]

        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(
            relative_position=relative_position,
            bidirectional=False,
            num_buckets=self.config.num_relative_pos,
            max_distance=128
        )

        if cross:
            values = self.enc_dec_attention_relative_attention_embedding(relative_position_bucket)
        else:
            values = self.self_attention_relative_attention_embedding(relative_position_bucket)
        values = values.transpose((2, 0, 1))[None, :, :, :]
        return values


class Transformer(nn.Module):
    config: TransformerConfig

    def setup(self):
        self.embedding_layer = EmbeddingLayer(config=self.config)
        self.encoder = Encoder(config=self.config)
        self.final_encoder_layer_norm = LayerNorm(config=self.config)
        self.decoder = Decoder(config=self.config)
        self.final_decoder_layer_norm = LayerNorm(config=self.config)
        self.lm_head = nn.Dense(self.config.vocab_size, use_bias=False)

    def __call__(self, x: Array, max_tokens: int, debug: bool) -> Array:
        x = jax.lax.stop_gradient(x)
        x_emb = self.embedding_layer(x)
        encoding = self.encoder(x_emb)
        encoding = self.final_encoder_layer_norm(encoding)
        print("encodings done!")
        outputs = jnp.array([[0]])
        # for _ in range(max_tokens):
        while outputs[:, -1] != 1 and len(outputs[0]) < max_tokens:
            print("started token")
            if debug:
                bkpt()
            decoder_inputs_embedding = self.embedding_layer(outputs)
            decoding = self.decoder(decoder_inputs_embedding, encoding)
            decoding = self.final_decoder_layer_norm(decoding)
            # print(f"decoding before rescaling for outputs: {outputs}\n{decoding[0][0][:20]}")
            decoding *= (self.config.d_model ** -0.5)
            next_logits = self.lm_head(decoding)
            next_token = jax.lax.argmax(next_logits, axis=2, index_dtype=int)
            outputs = jax.lax.concatenate([outputs, next_token], dimension=1)
            print("finished token!")
        return outputs