#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
import pickle5 as pickle
import numpy as np
import time
from aux_train_tf import HybridTransformer, create_masks

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#load the data from pre-processing
train_path = 'train_set_streams.pickle'
test_path = 'test_set_streams.pickle'


with open(train_path, 'rb') as handle:
    trainSet = pickle.load(handle)

with open(test_path, 'rb') as handle:
    testSet = pickle.load(handle)

  
#train 
enc_inputO_train = np.int64(np.stack(trainSet['EncoderO_Input'])) #encoder Onset input
enc_inputG_train = np.int64(np.stack(trainSet['EncoderG_Input'])) #encoder Group input
enc_inputT_train = np.int64(np.stack(trainSet['EncoderT_Input'])) #encoder Type input
enc_inputD_train = np.int64(np.stack(trainSet['EncoderD_Input'])) #encoder Duration input
enc_inputV_train = np.int64(np.stack(trainSet['EncoderV_Input'])) #encoder Valueinput
dec_inputO_train = np.int64(np.stack(trainSet['DecoderO_Input'])) #decoder onset stream
dec_outputO_train = np.int64(np.stack(trainSet['DecoderO_Output']))
dec_inputD_train = np.int64(np.stack(trainSet['DecoderD_Input'])) #decoder drums stream
dec_outputD_train = np.int64(np.stack(trainSet['DecoderD_Output']))
#validation
enc_inputO_val = np.int64(np.stack(testSet['EncoderO_Input']))
enc_inputG_val = np.int64(np.stack(testSet['EncoderG_Input']))
enc_inputT_val = np.int64(np.stack(testSet['EncoderT_Input']))
enc_inputD_val = np.int64(np.stack(testSet['EncoderD_Input']))
enc_inputV_val = np.int64(np.stack(testSet['EncoderV_Input']))
dec_inputO_val = np.int64(np.stack(testSet['DecoderO_Input']))
dec_outputO_val = np.int64(np.stack(testSet['DecoderO_Output']))
dec_inputD_val = np.int64(np.stack(testSet['DecoderD_Input']))
dec_outputD_val = np.int64(np.stack(testSet['DecoderD_Output']))


#prepare datasets
BUFFER_SIZE = len(enc_inputO_train)
BUFFER_SIZE_EVAL = len(enc_inputO_val)
BATCH_SIZE = 32 #set batch size
steps_per_epoch = BUFFER_SIZE//BATCH_SIZE
steps_per_epoch_eval = BUFFER_SIZE_EVAL//BATCH_SIZE

#create training and evaluation tf dataset
dataset = tf.data.Dataset.from_tensor_slices((enc_inputO_train, enc_inputG_train, enc_inputT_train, 
                                              enc_inputD_train, enc_inputV_train,
                                              dec_inputO_train, dec_outputO_train,
                                              dec_inputD_train, dec_outputD_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

dataset_eval = tf.data.Dataset.from_tensor_slices((enc_inputO_val, enc_inputG_val, enc_inputT_val, 
                                                   enc_inputD_val, enc_inputV_val,
                                                   dec_inputO_val, dec_outputO_val,
                                                   dec_inputD_val, dec_outputD_val)).shuffle(BUFFER_SIZE_EVAL)
dataset_eval = dataset_eval.batch(BATCH_SIZE, drop_remainder=True)


#set transformer hyper parameters
num_layers = 4  #attention layers
#Embeddings
d_model_enc1 = 64 #Encoder Onset
d_model_enc2 = 16 #Encoder Instrument
d_model_enc3 = 32 #Encoder Type
d_model_enc4 = 64 #Encoder Duration 
d_model_enc5 = 64 #Encoder Value 

d_model_dec1 = 96 #Decoder Onset Embedding
d_model_dec2 = 96 #Decoder Drums Embedding

units = 1024 #for Dense Layers and BLSTM Encoder
num_heads = 8 #
dropout_rate = 0.3
epochs = 200 
#vocab sizes
enc_vocab_onsets = 31
enc_vocab_group = 5
enc_vocab_type = 7
enc_vocab_dur = 40
enc_vocab_value = 33

dec_vocab_onsets = 31
dec_vocab_drums = 16
#sequence lengths
enc_seq_length = 597
dec_seq_length = 545
#for relative attention half or full window
rel_dec_seq = dec_seq_length

model = HybridTransformer(num_layers=num_layers, d_model_enc1=d_model_enc1, d_model_enc2=d_model_enc2, 
                          d_model_enc3=d_model_enc3, d_model_enc4=d_model_enc4, d_model_enc5=d_model_enc5, 
                          d_model_dec1=d_model_dec1, d_model_dec2=d_model_dec2, num_heads=num_heads,
                          dff=units, input_vocab1=enc_vocab_onsets+1, input_vocab2=enc_vocab_group+1, 
                          input_vocab3=enc_vocab_type+1, input_vocab4=enc_vocab_dur+1, 
                          input_vocab5=enc_vocab_value+1, target_vocab1=dec_vocab_onsets+1, 
                          target_vocab2=dec_vocab_drums+1,pe_target=dec_seq_length, 
                          mode_choice='relative', #change to multihead for vanilla attentio mechanism
                          max_rel_pos_tar=rel_dec_seq, rate=dropout_rate)



#Set Optimizers and Loss Function
optimizer = tf.keras.optimizers.Adam(0.0005, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False, reduction='none')


def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


def accuracy_function(real, pred):
  accuracies = tf.equal(real, tf.argmax(pred, axis=2))

  mask = tf.math.logical_not(tf.math.equal(real, 0))
  accuracies = tf.math.logical_and(mask, accuracies)

  accuracies = tf.cast(accuracies, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

#Set TF Metrics
train_loss_onsets = tf.keras.metrics.Mean(name='train_loss_onsets')
train_accuracy_onsets = tf.keras.metrics.Mean(name='train_accuracy_onsets')
train_loss_drums = tf.keras.metrics.Mean(name='train_loss_drums')
train_accuracy_drums = tf.keras.metrics.Mean(name='train_accuracy_drums')

val_loss_onsets = tf.keras.metrics.Mean(name='val_loss_onsets')
val_accuracy_onsets = tf.keras.metrics.Mean(name='val_accuracy_onsets')
val_loss_drums = tf.keras.metrics.Mean(name='val_loss_drums')
val_accuracy_drums = tf.keras.metrics.Mean(name='val_accuracy_drums')


#Set Checkpoints

checkpoint_path = './checkpoints/'

ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print('Latest checkpoint restored!!')
  
  

# Set input signatures

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64)
]

val_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64)
]


'''Training and Validation functions'''
@tf.function(input_signature=train_step_signature)
def train_step(inp1, inp2, inp3, inp4, inp5, tar_inp1, tar_real1, 
               tar_inp2, tar_real2):

  _, combined_mask, dec_padding_mask = create_masks(inp1, tar_inp1)

  with tf.GradientTape() as tape:
    preds1, preds2, _ = model(inp1, inp2, inp3, inp4, inp5, 
                              tar_inp1, tar_inp2,
                              True,
                              combined_mask,
                              dec_padding_mask)
    
    loss1 = loss_function(tar_real1, preds1)
    loss2 = loss_function(tar_real2, preds2)
    loss = 0.5*loss1+0.5*loss2 #equal loss

  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  
  acc1 = accuracy_function(tar_real1, preds1)
  acc2 = accuracy_function(tar_real2, preds2)

  train_loss_onsets(loss1)
  train_loss_drums(loss2)
  train_accuracy_onsets(acc1)
  train_accuracy_drums(acc2)
  
  
  
@tf.function(input_signature=val_step_signature)
def val_step(inp1, inp2, inp3, inp4, inp5, tar_inp1, tar_real1, 
               tar_inp2, tar_real2):


  _, combined_mask, dec_padding_mask = create_masks(inp1, tar_inp1)

  preds1, preds2, _ = model(inp1, inp2, inp3, inp4, inp5, 
                            tar_inp1, tar_inp2,
                            False, #change?
                            combined_mask,
                            dec_padding_mask)
  
  loss1 = loss_function(tar_real1, preds1)
  loss2 = loss_function(tar_real2, preds2)
  
  acc1 = accuracy_function(tar_real1, preds1)
  acc2 = accuracy_function(tar_real2, preds2)

  val_loss_onsets(loss1)
  val_loss_drums(loss2)
  val_accuracy_onsets(acc1)
  val_accuracy_drums(acc2)
  
  
 
"""START TRAINING"""
patience = 0
curr_loss = 99.99    
for epoch in range(epochs):
  start = time.time()

  train_loss_onsets.reset_states()
  train_accuracy_onsets.reset_states()
  train_loss_drums.reset_states()
  train_accuracy_drums.reset_states()
  
  print(f'Epoch {epoch + 1}')
  print('----')
  for (batch, (inp1, inp2, inp3, inp4, inp5, tar_inp1, tar_real1, 
               tar_inp2, tar_real2)) in enumerate(dataset.take(steps_per_epoch)):
    train_step(inp1, inp2, inp3, inp4, inp5, 
               tar_inp1, tar_real1, tar_inp2, tar_real2)


    if batch % 50 == 0:
      print(f'Batch {batch}')
      print(f'Onset Loss {train_loss_onsets.result():.4f} -- Onset Accuracy {train_accuracy_onsets.result():.4f}')
      print(f'Drums Loss {train_loss_drums.result():.4f} -- Drums Accuracy {train_accuracy_drums.result():.4f}')

  print('----')
  print(f'Onset Loss {train_loss_onsets.result():.4f} -- Onset Accuracy {train_accuracy_onsets.result():.4f}')
  print(f'Drums Loss {train_loss_drums.result():.4f} -- Drums Accuracy {train_accuracy_drums.result():.4f}')
  
  
  print('Evaluating...')

  val_loss_onsets.reset_states()
  val_accuracy_onsets.reset_states()  
  val_loss_drums.reset_states()
  val_accuracy_drums.reset_states()  
  
  for (batch, (inp1, inp2, inp3, inp4, inp5, tar_inp1, tar_real1, 
               tar_inp2, tar_real2)) in enumerate(dataset_eval.take(steps_per_epoch_eval)):
    val_step(inp1, inp2, inp3, inp4, inp5, 
             tar_inp1, tar_real1, tar_inp2, tar_real2)
  
  print('----')
  print(f'Validation Onset Loss {val_loss_onsets.result():.4f} -- Onset Accuracy {val_accuracy_onsets.result():.4f}')  
  print(f'Validation Drums Loss {val_loss_drums.result():.4f} -- Drums Accuracy {val_accuracy_drums.result():.4f}')  
  
  val_loss = np.round((0.5*val_loss_onsets.result().numpy() + 0.5*val_loss_drums.result().numpy()), decimals = 5) #change weights
  print('Overall weighted Validation Loss: ', val_loss)
  
  '''EARLY STOP MECHANISM'''
  if curr_loss > val_loss:
    #save checkpoint
    print('Checkpoint saved.')
    patience = 0
    save_path = ckpt_manager.save()
    curr_loss = val_loss
    
  else:
      print('No validation loss improvement.')
      patience += 1
      
  print(f'Time taken for this epoch: {time.time() - start:.2f} secs\n')    
  print('*******************************')
      
  if patience > 5:
      print('Terminating the training.')
      break

  