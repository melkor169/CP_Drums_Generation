#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""


import numpy as np
import tensorflow as tf


def get_angles(pos, i, d_model):

  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))

  return pos * angle_rates

 

def positional_encoding(position, d_model):

  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)


 

def create_padding_mask(seq, data_format = "series"):

    #seq (batch, len) or (batch, len, 2)
    if data_format == "series":
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    elif data_format == "parallel":
        new_seq = seq[:,:,0]
        seq = tf.cast(tf.math.equal(new_seq, 0), tf.float32)

    # print("padding_mask_shape",seq.shape)

    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


 

def create_look_ahead_mask(size):

    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    #print("look ahead mask shape", mask.shape)

    return mask  # (seq_len, seq_len)

 

def create_masks(inp, tar, data_format = "series"):

    enc_padding_mask = create_padding_mask(inp,data_format)
    dec_padding_mask = create_padding_mask(inp,data_format)

   
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar, data_format)

    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask

 

def point_wise_feed_forward_network(d_model, dff):

    return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))  # batch, h, seq, dh
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output, attention_weights

def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights

class RelativeGlobalAttention(tf.keras.layers.Layer):

    def __init__(self, d_model=256,num_heads=4, max_relative_position=2048):

        super(RelativeGlobalAttention, self).__init__() #add this?

        self.max_relative_position = max_relative_position
        self.num_heads = num_heads
        self.d_model = d_model

        assert self.d_model % self.num_heads == 0

        self.depth = self.d_model // self.num_heads
        self.Wq = tf.keras.layers.Dense(self.d_model)
        self.Wk = tf.keras.layers.Dense(self.d_model)
        self.Wv = tf.keras.layers.Dense(self.d_model)
        self.relative_embeddings = None
        self.dense = tf.keras.layers.Dense(self.d_model)


    def split_heads(self, x, batch_size):

        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def _get_left_embedding(self, length):

        initializer_stddev = self.depth**-0.5
        embedding_shape = (self.max_relative_position, self.depth)
        #how to define variable tf2

        if self.relative_embeddings is None:

            self.relative_embeddings = tf.Variable(name="relative_embedding", 
                    initial_value=tf.random_normal_initializer(stddev=initializer_stddev)(shape=embedding_shape))

        pad_length = tf.maximum(length - self.max_relative_position, 0)
        padded_relative_embeddings = tf.pad(self.relative_embeddings, [[pad_length, 0], [0, 0]])
        start_slice_position = tf.maximum(self.max_relative_position - length, 0) #should be useless

        sliced_relative_embeddings = tf.slice( padded_relative_embeddings, [start_slice_position, 0], [length, -1]) #should be useless

        return sliced_relative_embeddings

    @staticmethod
    def _qe_masking(qe):

        mask = tf.sequence_mask(
            tf.range(tf.shape(qe)[-1] -1, tf.shape(qe)[-1] - tf.shape(qe)[-2] -1, -1), tf.shape(qe)[-1])
        mask = tf.logical_not(mask)
        mask = tf.cast(mask, tf.float32)
        
        return mask * qe

    def _skewing(self, QE, len_k, len_q):

        batch, heads, length = tf.shape(QE)[0], tf.shape(QE)[1], tf.shape(QE)[2]
        padded = tf.pad(QE, [[0, 0], [0,0], [0, 0], [1, 0]])
        reshaped = tf.reshape(padded, shape=[batch, heads, length+1, length])
        Srel = reshaped[:, :, 1:, :]

        if len_k > len_q:
            Srel = tf.pad(Srel, [[0,0], [0,0], [0,0], [0, len_k-len_q]])

        elif len_k < len_q:
            Srel = Srel[:,:,:,:len_k]

        return Srel

    def relative_global_attn(self, q, k, v, mask):
        # print(f"q shape: {q.shape}, k shape {k.shape}")
        logits = tf.matmul(q, k, transpose_b=True)
        len_k = tf.shape(k)[2]
        len_q = tf.shape(q)[2]

        E = self._get_left_embedding(len_q) # in the t2t version it uses len_k, but it assumes len_q == len_k
        QE = tf.einsum('bhld,md->bhlm', q, E)
        # print(f"E shape: {E.shape}, E: {E}, QE shape:{QE.shape}, QE:{QE}")
        QE = self._qe_masking(QE)#check this!

        Srel = self._skewing(QE, len_k, len_q)
        # print(f"Srel shape:{Srel.shape}, Srel:{Srel}")
        logits += Srel
        logits = logits / tf.math.sqrt(tf.cast(self.depth, tf.float32)) #why??
        if mask is not None:
            logits += (tf.cast(mask, tf.float32) * -1e9)

        attention_weights = tf.nn.softmax(logits)
        attention = tf.matmul(attention_weights, v)

        return attention, attention_weights
 
    def call(self, v, k, q, mask=None):

        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)
        
        batch_size = tf.shape(q)[0]

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        attention, attention_weights = self.relative_global_attn(q, k, v, mask)
        out = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(out, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)

        return output, attention_weights

 
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, max_relative_position=2048, mode_choice = None):
        super(DecoderLayer, self).__init__()

        if mode_choice == 'relative': #vtrg: vanilla_transformer_relative_glob; vtg: vanilla_transformer_glob
            self.attn1 = RelativeGlobalAttention(d_model, num_heads, max_relative_position)
            self.attn2 = RelativeGlobalAttention(d_model, num_heads, max_relative_position)

        elif mode_choice =='multihead': #vtg: vanilla_transformer_glob
            self.attn1 = MultiHeadAttention(d_model, num_heads)
            self.attn2 = MultiHeadAttention(d_model, num_heads)        

        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
            look_ahead_mask, padding_mask):

        # enc_output.shape == (batch_size, input_seq_len, d_model)
        attn1, attn_weights_block1 = self.attn1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        attn2, attn_weights_block2 = self.attn2(enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2



class WordDecoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model1, d_model2, num_heads, dff, target_vocab1, target_vocab2,
               maximum_position_encoding, mode_choice, max_rel_pos = 256, rate=0.1):
    super(WordDecoder, self).__init__()

    self.d_model1 = d_model1 #for 2 inputs
    self.d_model2 = d_model2
    self.d_model = d_model1+d_model2 #the overall after concat
    self.num_layers = num_layers

    self.embedding1 = tf.keras.layers.Embedding(target_vocab1, d_model1)
    self.embedding2 = tf.keras.layers.Embedding(target_vocab2, d_model2)
    self.embed_inp = tf.keras.layers.Dense(self.d_model, activation='linear')
    self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

    self.dec_layers = [DecoderLayer(self.d_model, num_heads, dff, rate, max_rel_pos, mode_choice)
                       for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x1, x2, enc_output, training,
           look_ahead_mask, padding_mask):

    seq_len = tf.shape(x1)[1] #same for both inputs
    attention_weights = {}

    x1 = self.embedding1(x1)  # (batch_size, target_seq_len, d_model1)
    x2 = self.embedding2(x2)  # (batch_size, target_seq_len, d_model2)
    x = tf.keras.layers.Concatenate()([x1,x2])
    x = self.embed_inp(x)
    
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                             look_ahead_mask, padding_mask)

      attention_weights[f'decoder_layer{i+1}_block1'] = block1
      attention_weights[f'decoder_layer{i+1}_block2'] = block2

    # x.shape == (batch_size, target_seq_len, d_model)
    return x, attention_weights


 

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)   
    

#BLSTM Enconder
class BLSTMEncoder(tf.keras.layers.Layer):
  def __init__(self, d_model1, d_model2, d_model3, d_model4, d_model5, 
               dff, input_vocab1, input_vocab2, input_vocab3, input_vocab4, 
               input_vocab5, rate=0.1):
    super(BLSTMEncoder, self).__init__()

    self.d_model1 = d_model1 #5 inputs
    self.d_model2 = d_model2
    self.d_model3 = d_model3
    self.d_model4 = d_model4
    self.d_model5 = d_model5
    self.d_model = d_model1+d_model2+d_model3+d_model4+d_model5 #the overall after concat
    
    self.units = int(dff/2)

    self.embedding1 = tf.keras.layers.Embedding(input_vocab1, d_model1, mask_zero=True)
    self.embedding2 = tf.keras.layers.Embedding(input_vocab2, d_model2, mask_zero=True)
    self.embedding3 = tf.keras.layers.Embedding(input_vocab3, d_model3, mask_zero=True)
    self.embedding4 = tf.keras.layers.Embedding(input_vocab4, d_model4, mask_zero=True)
    self.embedding5 = tf.keras.layers.Embedding(input_vocab5, d_model5, mask_zero=True)
    
    self.embed_inp = tf.keras.layers.Dense(self.d_model, activation='linear')
    
    self.BLSTM1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.units,return_sequences=True))
    self.dropout1 = tf.keras.layers.Dropout(rate)
    
    self.BLSTM2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.units,return_sequences=True))
    self.dropout2 = tf.keras.layers.Dropout(rate)
    
    self.BLSTM3 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.units,return_sequences=True))
    self.dropout3 = tf.keras.layers.Dropout(rate)

  def call(self, x1, x2, x3, x4, x5, training):

    # adding embedding and position encoding.
    x1 = self.embedding1(x1)  # (batch_size, input_seq_len, d_model1)
    x2 = self.embedding2(x2)  # (batch_size, input_seq_len, d_model2)
    x3 = self.embedding3(x3)  # (batch_size, input_seq_len, d_model3)
    x4 = self.embedding4(x4)  # (batch_size, input_seq_len, d_model4)
    x5 = self.embedding5(x5)  # (batch_size, input_seq_len, d_model5)
    
    x = tf.keras.layers.Concatenate()([x1,x2,x3,x4,x5])
    x = self.embed_inp(x)
    
    x = self.BLSTM1(x)
    x = self.dropout1(x, training=training)
    
    x = self.BLSTM2(x)
    x = self.dropout2(x, training=training)
    
    x = self.BLSTM3(x)
    x = self.dropout3(x, training=training)


    return x  # (batch_size, input_seq_len, dff)


#Model used in the paper
class HybridTransformer(tf.keras.Model):
  def __init__(self, num_layers, d_model_enc1, d_model_enc2, d_model_enc3, d_model_enc4,
               d_model_enc5, d_model_dec1, d_model_dec2, num_heads, dff, input_vocab1, 
               input_vocab2, input_vocab3, input_vocab4, input_vocab5, target_vocab1, 
               target_vocab2, pe_target, mode_choice, max_rel_pos_tar, rate=0.1):
    super(HybridTransformer, self).__init__()

    self.encoder = BLSTMEncoder(d_model_enc1, d_model_enc2, d_model_enc3, d_model_enc4, 
                                d_model_enc5, dff, input_vocab1, input_vocab2, 
                                input_vocab3, input_vocab4, input_vocab5, rate)

    self.decoder = WordDecoder(num_layers, d_model_dec1, d_model_dec2, num_heads, dff, 
                           target_vocab1, target_vocab2, pe_target, mode_choice, max_rel_pos_tar, rate)

    self.final_layer_tar1 = tf.keras.layers.Dense(target_vocab1, activation='softmax') #2 outs
    self.final_layer_tar2 = tf.keras.layers.Dense(target_vocab2, activation='softmax')

  def call(self, inp1, inp2, inp3, inp4, inp5, 
           tar1, tar2, training, look_ahead_mask, dec_padding_mask):

    enc_output = self.encoder(inp1, inp2, inp3, inp4, inp5, training)  # (batch_size, inp_seq_len, dff)

    # dec_output.shape == (batch_size, tar_seq_len, d_model)
    dec_output, attention_weights = self.decoder(
        tar1, tar2, enc_output, training, look_ahead_mask, dec_padding_mask)

    final_output1 = self.final_layer_tar1(dec_output)  # (batch_size, tar_seq_len, target_vocab1)
    final_output2 = self.final_layer_tar2(dec_output)  # (batch_size, tar_seq_len, target_vocab2)

    return final_output1, final_output2, attention_weights

