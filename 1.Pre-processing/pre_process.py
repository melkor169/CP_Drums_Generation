#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import pickle #for older python versions you may need pickle5
import pretty_midi as pm
from glob import glob
from copy import deepcopy
from aux_files import check_pm_instrs, merge_pm, get_pianorolls,\
    get_reduced, get_event_based_rep, merge_bars_fes,\
    break_phrase, create_enc_dec
  
        
    


"""GLOBAL VARIABLES"""
max_phrase_lgt = 16 # num of bars of phrase
cnt = 0 #for counting errors

"""1. Get all full version MIDI files"""          

#get lakh midis
inp_path = glob('./dataset/*/*/*.mid')



"""2a. Get attributes and break phrases"""
all_tracks = [] #list to store all info


for trk in inp_path:
    #get the MSD ID and check if desired instrs exist
    #msd_id = trk.split('/')[3] #For Mac,Linux
    msd_id = trk.split('\\')[2] #For Windows
    print('Parsing ', msd_id, ' ...')
    #get all bars info
    pm_data = pm.PrettyMIDI(trk)
    is_ok = check_pm_instrs(pm_data)
    if is_ok: #if guitar bass and drums instrs exist
        try: 
            #remove duplicate channels, merge drum tracks and get track idxs for guitar,bass and drums
            pm_data, g_trks, b_trks, d_trks = merge_pm(pm_data)
            #get the PRs for every instrument for every bar along with general info
            allBars_info, _ = get_pianorolls(pm_data)   
            #get reduced midi file fron the original
            trk_reduced = get_reduced(deepcopy(pm_data), allBars_info, g_trks, b_trks, d_trks, msd_id)
            #get event-based representation
            allIns_ev = get_event_based_rep(trk_reduced, allBars_info)
            #get pianorolls for the reduced version
            pm_red = pm.PrettyMIDI(trk_reduced)
            _, allBars_pRs = get_pianorolls(pm_red)  
            #merge allBars info with EVs
            allBars_info = merge_bars_fes(allBars_info, allIns_ev, allBars_pRs)
            #store it to the super list
            all_tracks.append(allBars_info)
   

        except Exception as e:         
            print('Aborted due to', e)
            cnt += 1
        
    else:
        print('Aborted due instrumentation')
        cnt += 1
        
        
print(cnt, 'tracks were aborted')


"""3.Creating Encoder Decoder parts using CP representations""" 
#5-CP Encoder and 2-CP Decoder streams

dataDict = {'All_events': [],
            'Encoder_Onset': [], 
            'Encoder_Group': [],
            'Encoder_Type': [],
            'Encoder_Duration': [],
            'Encoder_Value': [],
            'Decoder_Onset': [],
            'Decoder_Drums': []}

for p in range(0,len(all_tracks)):
    
    allBars_info  = all_tracks[p]
    #break the allBars_info if it is more than the threshold
    if len(allBars_info) > max_phrase_lgt:
       allBarsDict = break_phrase(allBars_info,max_phrase_lgt) 
       for _, pp in allBarsDict.items():
            #save the events
            dataDict['All_events'].append(pp)  
            #save the parallel events
            Enc_Onset, Enc_Group, Enc_Type, Enc_Duration, Enc_Value, Dec_Onsets, Dec_Drums = create_enc_dec(pp)
            #store it
            dataDict['Encoder_Onset'].append(Enc_Onset)
            dataDict['Encoder_Group'].append(Enc_Group)
            dataDict['Encoder_Type'].append(Enc_Type)
            dataDict['Encoder_Duration'].append(Enc_Duration)
            dataDict['Encoder_Value'].append(Enc_Value)
            dataDict['Decoder_Onset'].append(Dec_Onsets)
            dataDict['Decoder_Drums'].append(Dec_Drums)
             
    else:
        #save the events
        dataDict['All_events'].append(allBars_info)  
        #save parallel series
        Enc_Onset, Enc_Group, Enc_Type, Enc_Duration, Enc_Value, Dec_Onsets, Dec_Drums = create_enc_dec(allBars_info)
        #store it
        dataDict['Encoder_Onset'].append(Enc_Onset)
        dataDict['Encoder_Group'].append(Enc_Group)
        dataDict['Encoder_Type'].append(Enc_Type)
        dataDict['Encoder_Duration'].append(Enc_Duration)
        dataDict['Encoder_Value'].append(Enc_Value)
        dataDict['Decoder_Onset'].append(Dec_Onsets)
        dataDict['Decoder_Drums'].append(Dec_Drums)
        
"""4. Save it and run post_process for converting to one-hot vectors"""

data_path = 'preprocessed_dataset.pickle'
# you may download the pre-processed dataset used in the paper
with open(data_path, 'wb') as handle:
    pickle.dump(dataDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
 