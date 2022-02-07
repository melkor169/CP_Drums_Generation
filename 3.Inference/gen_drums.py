#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" #for GPU inference
from glob import glob
import pickle
import pretty_midi as pm
from drums_utils import drums_trans_ev_model_tf, generate_drums_ev_trans_tf,\
    create_pp_instance, get_pianorolls, get_event_based_rep, merge_bars_fes,\
    create_enc, create_onehot_enc


'''load Encoders pickle for onehotencoders'''

#encoders pickle is created during pre-processing
encoders_trans = './aux_files/drums_encoders_cp.pickle'

    
with open(encoders_trans, 'rb') as handle:
    TransEncoders = pickle.load(handle)
#[EncOnset, EncGroup, EncType, EncDur, EncValue,
# DecOnset, DecDrums]


'''Load Inference Transformer. You may download pre-trained model based 
on the paper. See instructions in ReadME.md'''
trans_drums_hb = drums_trans_ev_model_tf(TransEncoders) 


'''Set Temperature'''
temperature = 0.9

'''Load MIDI files with Guitar (1st) and Bass (2nd). See examples in midi_in folder'''
'''max 16 bars'''
#input folder
inp_path = glob('./midi_in/*.mid')
#output folder
midi_out = './'

for trk in inp_path:
    #open with PrettyMIDI
    pm_data = pm.PrettyMIDI(trk)
    #get name
    trk_name = trk.split('\\')[-1][:-4] #you may change it depending your OS
    try:
        print('Generating..', trk_name)
        #get midi info and pianorolls
        allBars_info, allBars_pRs = get_pianorolls(pm_data)   
        #get event-based representation
        allIns_ev = get_event_based_rep(trk, allBars_info)
        #merge allBars info with EVs
        allBars_info = merge_bars_fes(allBars_info, allIns_ev, allBars_pRs)
        #create the Encoder
        Enc_Onset, Enc_Group, Enc_Type, Enc_Duration, Enc_Value = create_enc(allBars_info)
        #convert the CP streams to one-hot. It may raise exceptions if an event has not be
        #found during the training (e.g. rare Time Signature, duration etc)
        Enc_InputO, Enc_InputG, Enc_InputT, Enc_InputD, Enc_InputV = create_onehot_enc(Enc_Onset, 
                          Enc_Group, Enc_Type, Enc_Duration, Enc_Value, TransEncoders)
        #call generation functions
        onsets_HB, drums_HB = generate_drums_ev_trans_tf(trans_drums_hb, TransEncoders, temperature, 
                            Enc_InputO, Enc_InputG, Enc_InputT, Enc_InputD, Enc_InputV)      
        #create pypianoroll instance
        pp_ins = create_pp_instance(allBars_info, onsets_HB, drums_HB)
        #save midi files
        pp_ins.write(midi_out+trk_name+'_with_Drums.mid')


    except Exception as e:         
       print('Aborted due to', e)
            
  




 
    


    

    
 