# Set up names for new weights
new_weights = {'embedding_layer.word_embedding_layer.weight': torch.Size([32128, 512]),
 'encoder.layers.0.self_attention.w_q.weight': torch.Size([512, 512]),
 'encoder.layers.0.self_attention.w_k.weight': torch.Size([512, 512]),
 'encoder.layers.0.self_attention.w_v.weight': torch.Size([512, 512]),
 'encoder.layers.0.self_attention.w_o.weight': torch.Size([512, 512]),
 'encoder.self_attention_relative_attention_embedding.weight': torch.Size([32, 8]),
 'encoder.layers.0.norm1.weight': torch.Size([512]),
 'encoder.layers.0.mlp.ff1.weight': torch.Size([2048, 512]),
 'encoder.layers.0.mlp.ff2.weight': torch.Size([512, 2048]),
 'encoder.layers.0.norm2.weight': torch.Size([512]),
 'encoder.layers.1.self_attention.w_q.weight': torch.Size([512, 512]),
 'encoder.layers.1.self_attention.w_k.weight': torch.Size([512, 512]),
 'encoder.layers.1.self_attention.w_v.weight': torch.Size([512, 512]),
 'encoder.layers.1.self_attention.w_o.weight': torch.Size([512, 512]),
 'encoder.layers.1.norm1.weight': torch.Size([512]),
 'encoder.layers.1.mlp.ff1.weight': torch.Size([2048, 512]),
 'encoder.layers.1.mlp.ff2.weight': torch.Size([512, 2048]),
 'encoder.layers.1.norm2.weight': torch.Size([512]),
 'encoder.layers.2.self_attention.w_q.weight': torch.Size([512, 512]),
 'encoder.layers.2.self_attention.w_k.weight': torch.Size([512, 512]),
 'encoder.layers.2.self_attention.w_v.weight': torch.Size([512, 512]),
 'encoder.layers.2.self_attention.w_o.weight': torch.Size([512, 512]),
 'encoder.layers.2.norm1.weight': torch.Size([512]),
 'encoder.layers.2.mlp.ff1.weight': torch.Size([2048, 512]),
 'encoder.layers.2.mlp.ff2.weight': torch.Size([512, 2048]),
 'encoder.layers.2.norm2.weight': torch.Size([512]),
 'encoder.layers.3.self_attention.w_q.weight': torch.Size([512, 512]),
 'encoder.layers.3.self_attention.w_k.weight': torch.Size([512, 512]),
 'encoder.layers.3.self_attention.w_v.weight': torch.Size([512, 512]),
 'encoder.layers.3.self_attention.w_o.weight': torch.Size([512, 512]),
 'encoder.layers.3.norm1.weight': torch.Size([512]),
 'encoder.layers.3.mlp.ff1.weight': torch.Size([2048, 512]),
 'encoder.layers.3.mlp.ff2.weight': torch.Size([512, 2048]),
 'encoder.layers.3.norm2.weight': torch.Size([512]),
 'encoder.layers.4.self_attention.w_q.weight': torch.Size([512, 512]),
 'encoder.layers.4.self_attention.w_k.weight': torch.Size([512, 512]),
 'encoder.layers.4.self_attention.w_v.weight': torch.Size([512, 512]),
 'encoder.layers.4.self_attention.w_o.weight': torch.Size([512, 512]),
 'encoder.layers.4.norm1.weight': torch.Size([512]),
 'encoder.layers.4.mlp.ff1.weight': torch.Size([2048, 512]),
 'encoder.layers.4.mlp.ff2.weight': torch.Size([512, 2048]),
 'encoder.layers.4.norm2.weight': torch.Size([512]),
 'encoder.layers.5.self_attention.w_q.weight': torch.Size([512, 512]),
 'encoder.layers.5.self_attention.w_k.weight': torch.Size([512, 512]),
 'encoder.layers.5.self_attention.w_v.weight': torch.Size([512, 512]),
 'encoder.layers.5.self_attention.w_o.weight': torch.Size([512, 512]),
 'encoder.layers.5.norm1.weight': torch.Size([512]),
 'encoder.layers.5.mlp.ff1.weight': torch.Size([2048, 512]),
 'encoder.layers.5.mlp.ff2.weight': torch.Size([512, 2048]),
 'encoder.layers.5.norm2.weight': torch.Size([512]),
 'final_encoder_layer_norm.weight': torch.Size([512]),
 'decoder.layers.0.self_attention.w_q.weight': torch.Size([512, 512]),
 'decoder.layers.0.self_attention.w_k.weight': torch.Size([512, 512]),
 'decoder.layers.0.self_attention.w_v.weight': torch.Size([512, 512]),
 'decoder.layers.0.self_attention.w_o.weight': torch.Size([512, 512]),
 'decoder.self_attention_relative_attention_embedding.weight': torch.Size([32, 8]),
 'decoder.layers.0.norm1.weight': torch.Size([512]),
 'decoder.layers.0.enc_dec_attention.w_q.weight': torch.Size([512, 512]),
 'decoder.layers.0.enc_dec_attention.w_k.weight': torch.Size([512, 512]),
 'decoder.layers.0.enc_dec_attention.w_v.weight': torch.Size([512, 512]),
 'decoder.layers.0.enc_dec_attention.w_o.weight': torch.Size([512, 512]),
 'decoder.enc_dec_attention_relative_attention_embedding.weight': torch.Size([32, 8]),
 'decoder.layers.0.norm2.weight': torch.Size([512]),
 'decoder.layers.0.mlp.ff1.weight': torch.Size([2048, 512]),
 'decoder.layers.0.mlp.ff2.weight': torch.Size([512, 2048]),
 'decoder.layers.0.norm3.weight': torch.Size([512]),
 'decoder.layers.1.self_attention.w_q.weight': torch.Size([512, 512]),
 'decoder.layers.1.self_attention.w_k.weight': torch.Size([512, 512]),
 'decoder.layers.1.self_attention.w_v.weight': torch.Size([512, 512]),
 'decoder.layers.1.self_attention.w_o.weight': torch.Size([512, 512]),
 'decoder.layers.1.norm1.weight': torch.Size([512]),
 'decoder.layers.1.enc_dec_attention.w_q.weight': torch.Size([512, 512]),
 'decoder.layers.1.enc_dec_attention.w_k.weight': torch.Size([512, 512]),
 'decoder.layers.1.enc_dec_attention.w_v.weight': torch.Size([512, 512]),
 'decoder.layers.1.enc_dec_attention.w_o.weight': torch.Size([512, 512]),
 'decoder.layers.1.norm2.weight': torch.Size([512]),
 'decoder.layers.1.mlp.ff1.weight': torch.Size([2048, 512]),
 'decoder.layers.1.mlp.ff2.weight': torch.Size([512, 2048]),
 'decoder.layers.1.norm3.weight': torch.Size([512]),
 'decoder.layers.2.self_attention.w_q.weight': torch.Size([512, 512]),
 'decoder.layers.2.self_attention.w_k.weight': torch.Size([512, 512]),
 'decoder.layers.2.self_attention.w_v.weight': torch.Size([512, 512]),
 'decoder.layers.2.self_attention.w_o.weight': torch.Size([512, 512]),
 'decoder.layers.2.norm1.weight': torch.Size([512]),
 'decoder.layers.2.enc_dec_attention.w_q.weight': torch.Size([512, 512]),
 'decoder.layers.2.enc_dec_attention.w_k.weight': torch.Size([512, 512]),
 'decoder.layers.2.enc_dec_attention.w_v.weight': torch.Size([512, 512]),
 'decoder.layers.2.enc_dec_attention.w_o.weight': torch.Size([512, 512]),
 'decoder.layers.2.norm2.weight': torch.Size([512]),
 'decoder.layers.2.mlp.ff1.weight': torch.Size([2048, 512]),
 'decoder.layers.2.mlp.ff2.weight': torch.Size([512, 2048]),
 'decoder.layers.2.norm3.weight': torch.Size([512]),
 'decoder.layers.3.self_attention.w_q.weight': torch.Size([512, 512]),
 'decoder.layers.3.self_attention.w_k.weight': torch.Size([512, 512]),
 'decoder.layers.3.self_attention.w_v.weight': torch.Size([512, 512]),
 'decoder.layers.3.self_attention.w_o.weight': torch.Size([512, 512]),
 'decoder.layers.3.norm1.weight': torch.Size([512]),
 'decoder.layers.3.enc_dec_attention.w_q.weight': torch.Size([512, 512]),
 'decoder.layers.3.enc_dec_attention.w_k.weight': torch.Size([512, 512]),
 'decoder.layers.3.enc_dec_attention.w_v.weight': torch.Size([512, 512]),
 'decoder.layers.3.enc_dec_attention.w_o.weight': torch.Size([512, 512]),
 'decoder.layers.3.norm2.weight': torch.Size([512]),
 'decoder.layers.3.mlp.ff1.weight': torch.Size([2048, 512]),
 'decoder.layers.3.mlp.ff2.weight': torch.Size([512, 2048]),
 'decoder.layers.3.norm3.weight': torch.Size([512]),
 'decoder.layers.4.self_attention.w_q.weight': torch.Size([512, 512]),
 'decoder.layers.4.self_attention.w_k.weight': torch.Size([512, 512]),
 'decoder.layers.4.self_attention.w_v.weight': torch.Size([512, 512]),
 'decoder.layers.4.self_attention.w_o.weight': torch.Size([512, 512]),
 'decoder.layers.4.norm1.weight': torch.Size([512]),
 'decoder.layers.4.enc_dec_attention.w_q.weight': torch.Size([512, 512]),
 'decoder.layers.4.enc_dec_attention.w_k.weight': torch.Size([512, 512]),
 'decoder.layers.4.enc_dec_attention.w_v.weight': torch.Size([512, 512]),
 'decoder.layers.4.enc_dec_attention.w_o.weight': torch.Size([512, 512]),
 'decoder.layers.4.norm2.weight': torch.Size([512]),
 'decoder.layers.4.mlp.ff1.weight': torch.Size([2048, 512]),
 'decoder.layers.4.mlp.ff2.weight': torch.Size([512, 2048]),
 'decoder.layers.4.norm3.weight': torch.Size([512]),
 'decoder.layers.5.self_attention.w_q.weight': torch.Size([512, 512]),
 'decoder.layers.5.self_attention.w_k.weight': torch.Size([512, 512]),
 'decoder.layers.5.self_attention.w_v.weight': torch.Size([512, 512]),
 'decoder.layers.5.self_attention.w_o.weight': torch.Size([512, 512]),
 'decoder.layers.5.norm1.weight': torch.Size([512]),
 'decoder.layers.5.enc_dec_attention.w_q.weight': torch.Size([512, 512]),
 'decoder.layers.5.enc_dec_attention.w_k.weight': torch.Size([512, 512]),
 'decoder.layers.5.enc_dec_attention.w_v.weight': torch.Size([512, 512]),
 'decoder.layers.5.enc_dec_attention.w_o.weight': torch.Size([512, 512]),
 'decoder.layers.5.norm2.weight': torch.Size([512]),
 'decoder.layers.5.mlp.ff1.weight': torch.Size([2048, 512]),
 'decoder.layers.5.mlp.ff2.weight': torch.Size([512, 2048]),
 'decoder.layers.5.norm3.weight': torch.Size([512]),
 'final_decoder_layer_norm.weight': torch.Size([512]),
 'lm_head.weight': torch.Size([32128, 512])
}

t5_small_weights['lm_head.weight'] = t5_small_weights['shared.weight']

# Check num of weight parameters
len(new_weights.keys()), len(t5_small_weights.keys())

from collections import OrderedDict
t5_modified_weights = OrderedDict()

for (kv0, kv1) in zip(t5_small_weights.items(), new_weights.items()):
    key0, value0 = kv0
    key1, value1 = kv1
    t5_modified_weights[key1] = value0.detach().clone()
    
def load_weights():
    """
    Call this function to load weights.
    """
    transformer.load_state_dict(t5_modified_weights)
