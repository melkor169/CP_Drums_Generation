#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import pickle #for older python versions you may need pickle5
import numpy as np
from random import shuffle
from aux_files import create_onehot_dict
  
        
    


"""1.Post process to onehot encoding dictionaries"""
#load_data
data_path = 'preprocessed_dataset.pickle'

with open(data_path, 'rb') as handle:
    fDict = pickle.load(handle)

#calculate occurences (vocab sizes) for each CP stream and max encoder-decoder 
#sequence lengths

max_enc_length = 0 
max_dec_length = 0 

allEncO_Occs = [] #Encoder Onset
allEncG_Occs = [] #Encoder Group
allEncT_Occs = [] #Encoder Type
allEncD_Occs = [] #Encoder Duration
allEncV_Occs = [] #Encoder Value


allDecO_Occs = [] #Decoder Onset
allDecD_Occs = [] #Decoder Drums

for k in range(0, len(fDict['Encoder_Onset'])):
    
    #get max seq_lengths
    if max_enc_length < len(fDict['Encoder_Onset'][k]):
        max_enc_length = len(fDict['Encoder_Onset'][k])
    if max_dec_length < len(fDict['Decoder_Onset'][k]):
        max_dec_length = len(fDict['Decoder_Onset'][k])
    
    #get allEncoder and Decoder events and store them to the lists
    allEncO_Occs.extend(list(set(fDict['Encoder_Onset'][k])))
    allEncG_Occs.extend(list(set(fDict['Encoder_Group'][k])))
    allEncT_Occs.extend(list(set(fDict['Encoder_Type'][k])))
    allEncD_Occs.extend(list(set(fDict['Encoder_Duration'][k])))
    allEncV_Occs.extend(list(set(fDict['Encoder_Value'][k])))
    
    allDecO_Occs.extend(list(set(fDict['Decoder_Onset'][k])))
    allDecD_Occs.extend(list(set(fDict['Decoder_Drums'][k])))
 
        
#Add in the vocabulories the EOS SOS flags Parallel
allEncO_Occs.extend(['sos','eos'])
allEncG_Occs.extend(['sos','eos'])
allEncT_Occs.extend(['sos','eos'])
allEncD_Occs.extend(['sos','eos'])
allEncV_Occs.extend(['sos','eos'])
allDecO_Occs.extend(['sos','eos'])
allDecD_Occs.extend(['sos','eos'])
#Create one-hot dictionaries
EncO_Encoder = create_onehot_dict(allEncO_Occs)
EncG_Encoder = create_onehot_dict(allEncG_Occs)
EncT_Encoder = create_onehot_dict(allEncT_Occs)
EncD_Encoder = create_onehot_dict(allEncD_Occs)
EncV_Encoder = create_onehot_dict(allEncV_Occs)
DecO_Encoder = create_onehot_dict(allDecO_Occs)
DecD_Encoder = create_onehot_dict(allDecD_Occs)

#vocabulory sizes
encO_vocab = EncO_Encoder.categories_[0].shape[0]  #31
encG_vocab = EncG_Encoder.categories_[0].shape[0]  #5
encT_vocab = EncT_Encoder.categories_[0].shape[0]  #7
encD_vocab = EncD_Encoder.categories_[0].shape[0]  #40
encV_vocab = EncV_Encoder.categories_[0].shape[0]  #33
decO_vocab = DecO_Encoder.categories_[0].shape[0]  #31
decD_vocab = DecD_Encoder.categories_[0].shape[0]  #16


#save the Encoders for the generation stage
encoders_path = 'drums_encoders_cp.pickle'
with open(encoders_path, 'wb') as handle:
    pickle.dump([EncO_Encoder, EncG_Encoder, EncT_Encoder, EncD_Encoder, 
                 EncV_Encoder, DecO_Encoder, DecD_Encoder], 
                handle, protocol=pickle.HIGHEST_PROTOCOL)
    

'''2. Transform the dictionaries to one-hot encodings and add padding'''

#set sequence length encoder decoder 
dec_seq_length = max_dec_length + 1 #for sos or eos #545
enc_seq_length = max_enc_length + 2 #for sos and eos indications #597



trainDict = {'All_Events': [],
              'EncoderO_Input': [],
             'EncoderG_Input' : [],
             'EncoderT_Input' : [],
             'EncoderD_Input' : [],
             'EncoderV_Input' : [],
            'DecoderO_Input': [],
            'DecoderO_Output': [],
            'DecoderD_Input': [],
            'DecoderD_Output': []}


for t in range(0, len(fDict['Encoder_Onset'])):
    #store All_Events for later use
    allEvents_seq = fDict['All_events'][t]
    trainDict['All_Events'].append(allEvents_seq)
    
    #prepare data for encoders decoders CP
    aEncO_seq = fDict['Encoder_Onset'][t]
    aEncI_seq = fDict['Encoder_Group'][t]
    aEncT_seq = fDict['Encoder_Type'][t]
    aEncD_seq = fDict['Encoder_Duration'][t]
    aEncV_seq = fDict['Encoder_Value'][t]
    
    aDecO_seq = fDict['Decoder_Onset'][t]
    aDecD_seq = fDict['Decoder_Drums'][t]
      
    pad_lgt_enc_P = enc_seq_length-len(aEncO_seq)-2 #calculate paddings
    pad_lgt_dec_P = dec_seq_length-len(aDecO_seq)-1 #same for both outputs

    
    '''Encoder'''
    Enc_pad_emb = np.array(pad_lgt_enc_P*[0])   
    
    Enc_InputO = EncO_Encoder.transform(np.array(['sos']+aEncO_seq+['eos']).reshape(-1, 1)).toarray()
    Enc_InputO = [np.where(r==1)[0][0] for r in Enc_InputO] #for embeddings
    Enc_InputO = [x+1 for x in Enc_InputO] #shift by one in order to have 0 as pad
    trainDict['EncoderO_Input'].append(np.concatenate((Enc_InputO,Enc_pad_emb), axis = 0))

    Enc_InputG = EncG_Encoder.transform(np.array(['sos']+aEncI_seq+['eos']).reshape(-1, 1)).toarray()
    Enc_InputG = [np.where(r==1)[0][0] for r in Enc_InputG] 
    Enc_InputG = [x+1 for x in Enc_InputG] 
    trainDict['EncoderG_Input'].append(np.concatenate((Enc_InputG,Enc_pad_emb), axis = 0))
 
    Enc_InputT = EncT_Encoder.transform(np.array(['sos']+aEncT_seq+['eos']).reshape(-1, 1)).toarray()
    Enc_InputT = [np.where(r==1)[0][0] for r in Enc_InputT] 
    Enc_InputT = [x+1 for x in Enc_InputT] 
    trainDict['EncoderT_Input'].append(np.concatenate((Enc_InputT,Enc_pad_emb), axis = 0))
    
    Enc_InputD = EncD_Encoder.transform(np.array(['sos']+aEncD_seq+['eos']).reshape(-1, 1)).toarray()
    Enc_InputD = [np.where(r==1)[0][0] for r in Enc_InputD] 
    Enc_InputD = [x+1 for x in Enc_InputD] 
    trainDict['EncoderD_Input'].append(np.concatenate((Enc_InputD,Enc_pad_emb), axis = 0))
    
    Enc_InputV = EncV_Encoder.transform(np.array(['sos']+aEncV_seq+['eos']).reshape(-1, 1)).toarray()
    Enc_InputV = [np.where(r==1)[0][0] for r in Enc_InputV] 
    Enc_InputV = [x+1 for x in Enc_InputV] 
    trainDict['EncoderV_Input'].append(np.concatenate((Enc_InputV,Enc_pad_emb), axis = 0))
    
    '''Decoder'''
    Dec_pad_emb = np.array(pad_lgt_dec_P*[0]) 
    
    Dec_InputO = DecO_Encoder.transform(np.array(['sos']+aDecO_seq).reshape(-1, 1)).toarray()
    Dec_InputO = [np.where(r==1)[0][0] for r in Dec_InputO] 
    Dec_InputO = [x+1 for x in Dec_InputO] 
    trainDict['DecoderO_Input'].append(np.concatenate((Dec_InputO,Dec_pad_emb), axis = 0)) 
    
    Dec_InputD = DecD_Encoder.transform(np.array(['sos']+aDecD_seq).reshape(-1, 1)).toarray()
    Dec_InputD = [np.where(r==1)[0][0] for r in Dec_InputD] 
    Dec_InputD = [x+1 for x in Dec_InputD]
    trainDict['DecoderD_Input'].append(np.concatenate((Dec_InputD,Dec_pad_emb), axis = 0)) 
    

    Dec_TfO = DecO_Encoder.transform(np.array(aDecO_seq+['eos']).reshape(-1, 1)).toarray()
    Dec_TfO = [np.where(r==1)[0][0] for r in Dec_TfO] 
    Dec_TfO = [x+1 for x in Dec_TfO] 
    trainDict['DecoderO_Output'].append(np.concatenate((Dec_TfO, Dec_pad_emb), axis = 0)) 
    
    Dec_TfD = DecD_Encoder.transform(np.array(aDecD_seq+['eos']).reshape(-1, 1)).toarray()
    Dec_TfD = [np.where(r==1)[0][0] for r in Dec_TfD] 
    Dec_TfD = [x+1 for x in Dec_TfD] 
    trainDict['DecoderD_Output'].append(np.concatenate((Dec_TfD, Dec_pad_emb), axis = 0)) 
    


'''Split the dataset to train test 85-15'''
index_shuf = list(range(len(trainDict['EncoderO_Input']))) #random shufling
shuffle(index_shuf)

trainSet = {'All_Events': [],
              'EncoderO_Input': [],
             'EncoderG_Input' : [],
             'EncoderT_Input' : [],
             'EncoderD_Input' : [],
             'EncoderV_Input' : [],
            'DecoderO_Input': [],
            'DecoderO_Output': [],
            'DecoderD_Input': [],
            'DecoderD_Output': []}

testSet = {'All_Events': [],
              'EncoderO_Input': [],
             'EncoderG_Input' : [],
             'EncoderT_Input' : [],
             'EncoderD_Input' : [],
             'EncoderV_Input' : [],
            'DecoderO_Input': [],
            'DecoderO_Output': [],
            'DecoderD_Input': [],
            'DecoderD_Output': []}


trIDXs = int(0.85*len(index_shuf))
for i in range(0,trIDXs):
    trainSet['All_Events'].append(trainDict['All_Events'][index_shuf[i]])
    trainSet['EncoderO_Input'].append(trainDict['EncoderO_Input'][index_shuf[i]])
    trainSet['EncoderG_Input'].append(trainDict['EncoderG_Input'][index_shuf[i]])
    trainSet['EncoderT_Input'].append(trainDict['EncoderT_Input'][index_shuf[i]])
    trainSet['EncoderD_Input'].append(trainDict['EncoderD_Input'][index_shuf[i]])
    trainSet['EncoderV_Input'].append(trainDict['EncoderV_Input'][index_shuf[i]])
    trainSet['DecoderO_Input'].append(trainDict['DecoderO_Input'][index_shuf[i]])
    trainSet['DecoderO_Output'].append(trainDict['DecoderO_Output'][index_shuf[i]])
    trainSet['DecoderD_Input'].append(trainDict['DecoderD_Input'][index_shuf[i]])
    trainSet['DecoderD_Output'].append(trainDict['DecoderD_Output'][index_shuf[i]])



for i in range(trIDXs,len(index_shuf)):
    testSet['All_Events'].append(trainDict['All_Events'][index_shuf[i]])
    testSet['EncoderO_Input'].append(trainDict['EncoderO_Input'][index_shuf[i]])
    testSet['EncoderG_Input'].append(trainDict['EncoderG_Input'][index_shuf[i]])
    testSet['EncoderT_Input'].append(trainDict['EncoderT_Input'][index_shuf[i]])
    testSet['EncoderD_Input'].append(trainDict['EncoderD_Input'][index_shuf[i]])
    testSet['EncoderV_Input'].append(trainDict['EncoderV_Input'][index_shuf[i]])
    testSet['DecoderO_Input'].append(trainDict['DecoderO_Input'][index_shuf[i]])
    testSet['DecoderO_Output'].append(trainDict['DecoderO_Output'][index_shuf[i]])
    testSet['DecoderD_Input'].append(trainDict['DecoderD_Input'][index_shuf[i]])
    testSet['DecoderD_Output'].append(trainDict['DecoderD_Output'][index_shuf[i]])


#save them
train_path = 'train_set_streams.pickle'
test_path = 'test_set_streams.pickle'

with open(train_path, 'wb') as handle:
    pickle.dump(trainSet, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open(test_path, 'wb') as handle:
    pickle.dump(testSet, handle, protocol=pickle.HIGHEST_PROTOCOL)  
