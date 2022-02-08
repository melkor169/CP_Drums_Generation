#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import tensorflow as tf
import numpy as np
import pretty_midi as pm
import pypianoroll as pp
import music21 as m21

from copy import deepcopy
from operator import itemgetter
from aux_files.aux_train_tf import HybridTransformer,  create_masks



"""GLOBAL VARIABLES"""
beat_res = 12 #beat resolution for pianoroll 1/16 triplet


def create_onehot_enc(Enc_Onset, Enc_Group, Enc_Type, Enc_Duration, Enc_Value, TransEncoders):
    
    
    Enc_InputO = TransEncoders[0].transform(np.array(['sos']+Enc_Onset+['eos']).reshape(-1, 1)).toarray()
    Enc_InputO = [np.where(r==1)[0][0] for r in Enc_InputO] #for embeddings
    Enc_InputO = [x+1 for x in Enc_InputO] #shift by one in order to have 0 as pad

    Enc_InputG = TransEncoders[1].transform(np.array(['sos']+Enc_Group+['eos']).reshape(-1, 1)).toarray()
    Enc_InputG = [np.where(r==1)[0][0] for r in Enc_InputG] 
    Enc_InputG = [x+1 for x in Enc_InputG] 
  
    Enc_InputT = TransEncoders[2].transform(np.array(['sos']+Enc_Type+['eos']).reshape(-1, 1)).toarray()
    Enc_InputT = [np.where(r==1)[0][0] for r in Enc_InputT] 
    Enc_InputT = [x+1 for x in Enc_InputT] 
   
    Enc_InputD = TransEncoders[3].transform(np.array(['sos']+Enc_Duration+['eos']).reshape(-1, 1)).toarray()
    Enc_InputD = [np.where(r==1)[0][0] for r in Enc_InputD] 
    Enc_InputD = [x+1 for x in Enc_InputD] 
    
    Enc_InputV = TransEncoders[4].transform(np.array(['sos']+Enc_Value+['eos']).reshape(-1, 1)).toarray()
    Enc_InputV = [np.where(r==1)[0][0] for r in Enc_InputV] 
    Enc_InputV = [x+1 for x in Enc_InputV] 
    
    
    return Enc_InputO, Enc_InputG, Enc_InputT, Enc_InputD, Enc_InputV
  

def create_enc(allBars_info):
    
    numOfBars = len(allBars_info)
    
    Enc_Onset = [] #parallel streams
    Enc_Instr = []
    Enc_Type = []
    Enc_Duration = [] 
    Enc_Value = []
    
    
    for i in range(0,numOfBars):
        '''Encoder'''
        '''first get the high level'''
        #8 static HL features where onset instr and duration are set high level
        Enc_Onset.extend(3*['Bar']) #begin of the bar
        Enc_Instr.extend(3*['high-level'])
        Enc_Duration.extend(3*['Bar'])
        #1.Bar
        Enc_Type.append('Bar')
        Enc_Value.append('Bar')
        #2.TempoID
        Enc_Type.append('Tempo')
        Enc_Value.append(get_tempoID(allBars_info[i]['tempo'])) 
        #3.TimeSignature
        Enc_Type.append('TimeSig')
        timesigID = 'ts_'+str(allBars_info[i]['numerator'])+'/'+str(allBars_info[i]['denominator'])
        Enc_Value.append(timesigID)
     
        '''building the guitar and bass encoder streams'''
        chordsEV = allBars_info[i]['Guitar_EV']
        bassEV = allBars_info[i]['Bass_EV']
        #add Guitar and Bass flags
        chordsEV = [x + ['Guitar'] for x in chordsEV]
        bassEV = [x + ['Bass'] for x in bassEV]
        allEVs = chordsEV +bassEV
        #sort them according to onsets
        allEVs = sorted(allEVs, key=itemgetter(1))
        #get them to the Encoder Streams
        for e in allEVs:
            Enc_Onset.append(e[1])
            Enc_Instr.append(e[3])
            Enc_Type.append(e[0])
            Enc_Duration.append(e[2])
            Enc_Value.append('NaN') #no value
        
        
    return Enc_Onset, Enc_Instr, Enc_Type, Enc_Duration, Enc_Value   



def sample(preds, temperature=1.0):
    '''
    @param preds: a np.array with the probabilities to all categories
    @param temperature: the temperature. Below 1.0 the network makes more "safe"
                        predictions
    @return: the index after the sampling
    '''
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
   
    return np.argmax(probas)  



def generate_drums_ev_trans_tf(trans_drums, TransEncoders, temperature, Enc_Onset,
                                Enc_Instr, Enc_Type, Enc_Dur, Enc_Value):
     
    dec_seq_length = 549
    
    #convert Encoder Inps to tensors
    
    Enc_Onset = tf.convert_to_tensor([Enc_Onset])
    Enc_Instr = tf.convert_to_tensor([Enc_Instr])
    Enc_Type = tf.convert_to_tensor([Enc_Type])
    Enc_Dur = tf.convert_to_tensor([Enc_Dur])
    Enc_Value = tf.convert_to_tensor([Enc_Value]) 
    
    #Prepare Decoder Inps 
    onset_sos_idx = int(np.where(TransEncoders[5].categories_[0] == 'sos')[0])+1
    onset_eos_idx = int(np.where(TransEncoders[5].categories_[0] == 'eos')[0])+1 
    
    drums_sos_idx = int(np.where(TransEncoders[6].categories_[0] == 'sos')[0])+1
    drums_eos_idx = int(np.where(TransEncoders[6].categories_[0] == 'eos')[0])+1 
    
    dec_onset_out = []
    dec_out_onset = [onset_sos_idx]
    dec_out_onset = tf.convert_to_tensor(dec_out_onset)
    dec_out_onset = tf.expand_dims(dec_out_onset, 0)
    
    dec_drums_out = []
    dec_out_drums = [drums_sos_idx]
    dec_out_drums = tf.convert_to_tensor(dec_out_drums)
    dec_out_drums = tf.expand_dims(dec_out_drums, 0)
    
    #start generating autoregressively WORD TRANSFORMER
    for _ in range(dec_seq_length):
       #masking
       
       _, combined_mask, dec_padding_mask = create_masks(Enc_Onset, dec_out_onset)
           
       preds_onsets, preds_drums, _ = trans_drums(Enc_Onset, Enc_Instr, Enc_Type, Enc_Dur,
                                                  Enc_Value, dec_out_onset, dec_out_drums, False, 
                                                   combined_mask, dec_padding_mask)
       
       #Onset out
       preds_onsets = preds_onsets[:, -1:, :].numpy() # (batch_size, 1, vocab_size) select the last word
       token_preds_onset = preds_onsets[-1,:].reshape(preds_onsets.shape[-1],)
       #apply diversity
       token_pred_on = sample(token_preds_onset, temperature)
       dec_onset_out.append(token_pred_on) #for numpy
       
       #Drums Out
       preds_drums = preds_drums[:, -1:, :].numpy() # (batch_size, 1, vocab_size) select the last word
       token_preds_drums = preds_drums[-1,:].reshape(preds_drums.shape[-1],)
       #apply diversity
       token_pred_dr = sample(token_preds_drums, temperature)
       dec_drums_out.append(token_pred_dr) #for numpy

       
       if token_pred_on == onset_eos_idx or token_pred_dr == drums_eos_idx:
           print('Generated', len(dec_onset_out), 'steps')
           break    #EOS
       else:
           #prepare for the next cycle
           #Onsets
           dec_out_onset = [onset_sos_idx]
           dec_out_onset.extend(dec_onset_out)
           dec_out_onset = tf.convert_to_tensor(dec_out_onset)
           dec_out_onset = tf.expand_dims(dec_out_onset, 0)
           #Drums
           dec_out_drums = [drums_sos_idx]
           dec_out_drums.extend(dec_drums_out)
           dec_out_drums = tf.convert_to_tensor(dec_out_drums)
           dec_out_drums = tf.expand_dims(dec_out_drums, 0)

           
    #convert outs to event based rep
    dec_onset_out = [x-1 for x in dec_onset_out] #shift by -1 to get the original
    onsetsEV = []
    for i in range(0,len(dec_onset_out)-1): #exclude eos
        onsetsEV.append(str(TransEncoders[5].categories_[0][dec_onset_out[i]]))
        
    dec_drums_out = [x-1 for x in dec_drums_out] #shift by -1 to get the original
    drumsEV = []
    for i in range(0,len(dec_drums_out)-1): #exclude eos
        drumsEV.append(str(TransEncoders[6].categories_[0][dec_drums_out[i]]))

    
    return onsetsEV, drumsEV



def get_tempoID(tempo):
    
    #define tempo values
    # tempo value pick
    if int(tempo) > 0 and int(tempo) <= 50: 
        tempo_ID = 'tempo_0'
    elif int(tempo) > 50 and int(tempo) <= 80:
        tempo_ID = 'tempo_1'
    elif int(tempo) > 80 and int(tempo) <= 110:
        tempo_ID = 'tempo_2'
    elif int(tempo) > 110 and int(tempo) <= 140:
        tempo_ID = 'tempo_3'
    elif int(tempo) > 140 and int(tempo) <= 170:
        tempo_ID = 'tempo_4'
    elif int(tempo) > 170:
        tempo_ID = 'tempo_5'
        
    return tempo_ID


                    

def rm_drum_ev(drumsEV):

    if 'Snare' in drumsEV and 'Snare_Stick' in drumsEV:
        drumsEV.remove('Snare_Stick')    

    if 'Closed_HH' in drumsEV and 'Open_HH' in drumsEV:
        drumsEV.remove('Closed_HH')

    if 'Ride_Bell' in drumsEV and 'Ride' in drumsEV:
        drumsEV.remove('Ride_Bell')      

    #return only 1(onset)+3 elements of drums
    return drumsEV[:4]         


def get_drum_element(k):
    
     if k in [35,36]: #Kick
          dru = 'Kick'    
     elif k in [38,40]: #Snare
          dru = 'Snare'
     elif k in [37,39]: #Snare Stick #31
          dru = 'Snare_Stick'
     elif k == 42: #Closed HH #and tambourine-cabasa 54,69
          dru = 'Closed_HH'
     elif k in [44,46]: #Open HH and pedal #and shaker-maracas 70,82
          dru = 'Open_HH'
     elif k in [51,59]: #Ride
          dru = 'Ride'
     elif k in [53,56]: #Ride Bell and Cowbell
          dru = 'Ride_Bell'
     elif k in [41,43,64]: #Tom3 (low) and tumba
          dru = 'Tom3'              
     elif k in [45,47,62,63]: #Tom2 (mid) and conga
          dru = 'Tom2'   
     elif k in [48,50,60,61]: #Tom1 (high) and bongoes
          dru = 'Tom1'
     elif k == 49: #Crash1
          dru = 'Crash1'
     elif k == 57: #Crash2
          dru = 'Crash2'
     elif k in [52,55]: #China Splash
          dru = 'China'
     else: #no event
          dru = 'NotFound'
          
     return dru



def fix_prs_lgts(allBars_pRs, pr_diff, bar_lgt, allBars_info):
    
    if pr_diff > 0: #add 0.0
        lgt_start = bar_lgt - pr_diff
        for i in range(lgt_start, bar_lgt):
            #get pr size
            pr_size = int((4*(allBars_info[i]['numerator']/allBars_info[i]['denominator']))/(1/beat_res))
            aPR =  np.zeros((pr_size, 128), dtype=bool)
            aPR_row = [aPR, aPR, aPR]
            allBars_pRs.append(aPR_row)
            
 
    elif pr_diff < 0: #remove the last
        allBars_pRs = allBars_pRs[:bar_lgt]
 
        
    return allBars_pRs


def merge_bars_fes(allBars_info, allIns_ev, allBars_pRs):

    bar_lgt = len(allBars_info)
    #fix PRs depending on length
        
    pr_diff = bar_lgt - len(allBars_pRs)
    if pr_diff !=0:
        allBars_pRs = fix_prs_lgts(allBars_pRs, pr_diff, bar_lgt, allBars_info)
        
    
    for i in range(0, len(allBars_info)):
        #set first the EV lists
        allBars_info[i]['Guitar_EV'] = allIns_ev[0][i]
        allBars_info[i]['Bass_EV'] = allIns_ev[1][i]
        #add pRs
        allBars_info[i]['Guitar_PR'] = allBars_pRs[i][0]
        allBars_info[i]['Bass_PR'] = allBars_pRs[i][1]

        
    return allBars_info    

def get_event_based_rep(ps, allBars_info):
    #get pseudo event-based representation on non-drum tracks and event-based on drums
    s = m21.converter.parse(ps)
 
    allIns_ev = [] #Guitar, Bass
    for i in range(0,len(s.parts)):
        p = s.parts[i]
        if p.hasVoices():
            p = p.flattenUnnecessaryVoices(force=True) #remove (if any) voices
        #p = p.notesAndRests.stream() #keep only note events
        p = p.notes.stream() #keep only note events without rests
        all_evs = convert_pseudo_events(p, allBars_info)
        allIns_ev.append(all_evs)
          
    
    return allIns_ev
         


def convert_pseudo_events(p, allBars_info):
    #take a part and make pseudo_events
    all_events = [] #events per bar
    beat_cnt = 0.0 #beat counter
    for i in range(0,len(allBars_info)):
        beat_start = beat_cnt
        beat_dur = allBars_info[i]['beat_dur']
        beat_end = beat_dur+beat_cnt
        #create event list for this measure
        e_bar = []     
        for e in p.getElementsByOffset(beat_start,beat_end, includeEndBoundary=False):
            #get they type of the event (chord, note or rest)
            item = e.classes[0]
            #get offset and dur
            offset = str(float(e.offset-beat_start))[:6]
            offset = 'on_'+offset
            aDur = e.duration.quarterLength
            if aDur > 0.0: #eliminate not valid events
                if aDur > beat_dur:
                    aDur = 'overlapping'
                else:
                    aDur = str(float(aDur))[:6]
                    aDur = 'du_'+aDur
                aEve = [item, offset, aDur] #create an event
                e_bar.append(aEve)
        #merge events with the same onsets to chord events with the longest dur
        e_bar_mg = merge_events(e_bar)
        #for the next loop
        beat_cnt = beat_end 
        all_events.append(e_bar_mg)
        
    return all_events

def merge_events(e_bar):
    
    e_bar_mg = [] #new list
    prev_onset = ''
    for e in e_bar:
        curr_onset = e[1]
        if curr_onset != prev_onset:
            #new event
            e_bar_mg.append(deepcopy(e))
            prev_onset = curr_onset
        else:
            #get the previous event from the list
            #change first the event type if nto
            if e_bar_mg[-1][0] == 'Note':
                e_bar_mg[-1][0] = 'Chord'
            #change the duration if it is longer
            #first check if it is not overalapping
            if e_bar_mg[-1][-1] != 'overlapping':
                prev_dur = float(e_bar_mg[-1][-1].split('_')[-1])
                curr_dur_str = e[-1]
                if curr_dur_str != 'overlapping':                 
                    curr_dur = float(e[-1].split('_')[-1])
                    if curr_dur > prev_dur:
                        e_bar_mg[-1][-1] = 'du_'+str(curr_dur)
                else:
                    e_bar_mg[-1][-1] = 'overlapping'
                
    return e_bar_mg


def drums_trans_ev_model_tf(TransEncoders):
    
    #get vocabs
    enc_vocab_onsets = len(TransEncoders[0].categories_[0])
    enc_vocab_instr = len(TransEncoders[1].categories_[0])
    enc_vocab_type = len(TransEncoders[2].categories_[0])
    enc_vocab_dur = len(TransEncoders[3].categories_[0])
    enc_vocab_value = len(TransEncoders[4].categories_[0])
    
    dec_vocab_onsets = len(TransEncoders[5].categories_[0])
    dec_vocab_drums = len(TransEncoders[6].categories_[0])
    
    #create the architecture first  
    num_layers = 4  #4
    d_model_enc1 = 64 #Encoder Onset
    d_model_enc2 = 16 #Encoder Group
    d_model_enc3 = 32 #Encoder Type
    d_model_enc4 = 64 #Encoder Duration 
    d_model_enc5 = 64 #Encoder Value 
    
    d_model_dec1 = 96 #Decoder Onset Embedding
    d_model_dec2 = 96 #Decoder Drums Embedding
    units = 1024 #for Dense
    num_heads = 8 #8
    dropout_rate = 0.3
    
    dec_seq_length = 545
    #for relative attention

    rel_dec_seq = dec_seq_length #int(dec_seq_length/2) 

    model = HybridTransformer(num_layers=num_layers, d_model_enc1=d_model_enc1, d_model_enc2=d_model_enc2, 
                          d_model_enc3=d_model_enc3, d_model_enc4=d_model_enc4, d_model_enc5=d_model_enc5, 
                          d_model_dec1=d_model_dec1, d_model_dec2=d_model_dec2, num_heads=num_heads,
                          dff=units, input_vocab1=enc_vocab_onsets+1, input_vocab2=enc_vocab_instr+1, 
                          input_vocab3=enc_vocab_type+1, input_vocab4=enc_vocab_dur+1, 
                          input_vocab5=enc_vocab_value+1, target_vocab1=dec_vocab_onsets+1, 
                          target_vocab2=dec_vocab_drums+1,pe_target=dec_seq_length, mode_choice='relative', 
                          max_rel_pos_tar=rel_dec_seq, rate=dropout_rate)
        
    checkpoint_path = './aux_files/checkpoints/'
    print('Loading Hybrid Music Transformer')

    #Set Optimizers and load checkpoints
    
    optimizer = tf.keras.optimizers.Adam(0.0005, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)
        
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)
    
    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
      ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
      print('Latest checkpoint restored!')
    
    return model

def get_pianorolls(pm_data):
    '''
    @param pm_data: the PrettyMIDI input instance of this phrase
    @return: a list with general musical info (allBars_info) for every bar and a 
    list with each node corresponds the bar containing the PianoRolls of every track
    '''

    #get time sig tempo changes and downbeats
    all_timesigs = pm_data.time_signature_changes
    all_downbeats = pm_data.get_downbeats() #reveals the number of bars
    all_tempos = pm_data.get_tempo_changes()
    ending_t = pm_data.get_end_time() #end of time
    #get the info regarding timesig tempo timing mapping for every bar
    allBars_info = calc_infoBars(all_downbeats,all_timesigs,all_tempos,ending_t)
    allBars_pRs = []
    for b in range(0,len(allBars_info)):
        beg = allBars_info[b]['time_start']
        end = allBars_info[b]['time_end']
        dur = end-beg
        numerator = allBars_info[b]['numerator']
        denominator = allBars_info[b]['denominator']
        #calculate the num of timesteps of this bar and the duration of each timestep
        timestep_dur, total_timesteps = arrange_timesteps(numerator, denominator, beat_res, dur)
        aBar_pp = []
        #get all the PRs for this bar
        for i in range(0, len(pm_data.instruments)):
            
            anInstr = deepcopy(pm_data.instruments[i])
            #first get the corresponding notes of this bar according to PrettyMIDI
            newInstr = get_instrument_notes(anInstr, beg, end)
            #then create a quantized PR
            inst_pp = get_pianoRoll(timestep_dur, total_timesteps, dur, newInstr)
            aBar_pp.append(inst_pp)
            
        allBars_pRs.append(aBar_pp)
        
    return allBars_info, allBars_pRs


def get_pianoRoll(timestep_dur, total_timesteps, dur, anInstrument):
    '''
    @param timestep_dur: the duration of each timestep
    @param total_timesteps: the total timesteps
    @param dur: the overall duration of this bar
    @anInstrument: the input instrument containing notes for this bar in PrettyMIDI instance
    @return: a quantized (to the next 1/16th) pianoroll for this bar
    '''
    
    #create a PianoRoll instance
    aPR =  np.zeros((total_timesteps, 128), dtype=bool)
    
    #create the onset range for every timestep
    onset_range = []
    #for the first timestep
    onset_range.append([0.0, timestep_dur/2])
    for i in range(1,total_timesteps):
        #get previous border
        prev_o = onset_range[i-1][-1]
        #add plus timestep_dur
        onset_range.append([prev_o, prev_o+timestep_dur])
        
    #create the offset range for every timestep
    offset_range = []
    for i in range(0,total_timesteps-1):
        offset_range.append([i*timestep_dur + timestep_dur/2, (i+1)*timestep_dur + timestep_dur/2])
    #for the last timestep
    prev_s = offset_range[-1][-1] 
    offset_range.append([prev_s, dur]) #dur is the offset of the bar
    
    #check if it is a drum component
    isDrum = anInstrument.is_drum
    
    for note in anInstrument.notes:
        #first get the onset
        onset = note.start
        #check if the onset is in range of the mapping
        try:
            ons_val = next(r for r in onset_range if onset >= r[0] and onset < r[1])
            #get the idx
            ons_idx = onset_range.index(ons_val)
            
            if isDrum == False: #get the offsets
                offset = note.end
                off_val = next(r for r in offset_range if offset > r[0] and offset <= r[1])
                off_idx = offset_range.index(off_val)
                #check if the duration is ok to be a valid note
                dur = offset-onset
                if dur >= timestep_dur/2:
                    #assing full length
                    aPR[ons_idx:off_idx+1 ,note.pitch] = True
            else:
                #assign only onset
                aPR[ons_idx ,note.pitch] = True
                    
        except StopIteration:
            #rejected note
            continue
                
                
    return aPR

def calc_infoBars(all_downbeats,all_timesigs,all_tempos,ending_t):
    '''
    @param all_downbeats: the starting time of each new bar according to PrettyMIDI
    @param all_timesigs: all the acquired time_signatures changes according to PrettyMIDI
    @param all_tempos: all the acquired tempo changes according to PrettyMIDI
    @param ending_t: gives the overall duration; the offset of the last note of the PrettyMIDI instance.
    @return: a list of dictionaries where each node is information for each bar regarding timesigs, tempos
    beat/time start/end timings.
    '''

    timeSigsOnsets = []
    for x in range(0,len(all_timesigs)):
        timeSigsOnsets.append(all_timesigs[x].time)
        
    #make list alltempos
    tempos_idxs = list(all_tempos[0])
    tempos_vals = list(all_tempos[1])
        
    curr_den = all_timesigs[0].denominator
    curr_num = all_timesigs[0].numerator
    allBars = []
    #initialize the first time sig and bar
    aBar = {}
    beat_cnt = 0.0
    aBar['bar'] = 1
    aBar['tempo'] = tempos_vals[0]
    aBar['denominator'] = curr_den
    aBar['numerator'] = curr_num
    aBar['beat_start'] = beat_cnt
    aBar['time_start'] = all_downbeats[0]
    aBar['time_end'] = all_downbeats[1]
    dur = (curr_num*4)/curr_den
    aBar['beat_dur'] = dur
    allBars.append(aBar)
    beat_cnt = beat_cnt + dur
    
    #check if there is another timesig
    for k in range(1,len(all_downbeats)-1):
        curr_time = all_downbeats[k]
        end_time = all_downbeats[k+1]
        if curr_time in timeSigsOnsets:
            idx = timeSigsOnsets.index(curr_time)
            curr_den = all_timesigs[idx].denominator
            curr_num = all_timesigs[idx].numerator
        #for tempos search between curr_time and end_time if anything occurs
        ktempos_idxs= [val for val in tempos_idxs if val >= curr_time and val <= end_time]
        if len(ktempos_idxs) == 0:
            #get the previous one
            ktempo = allBars[k-1]['tempo']
        else:
            #get the last
            ktempo = tempos_vals[tempos_idxs.index(ktempos_idxs[-1])]   
        
        kBar = {}
        kBar['bar'] = k+1
        kBar['tempo'] = ktempo
        kBar['denominator'] = curr_den
        kBar['numerator'] = curr_num
        kBar['beat_start'] = beat_cnt
        kBar['time_start'] = curr_time
        kBar['time_end'] = end_time
        dur = (curr_num*4)/curr_den
        kBar['beat_dur'] = dur
        allBars.append(kBar)
        beat_cnt = beat_cnt + dur
        
    #for the last
    lBar = {}
    curr_time = all_downbeats[-1]
    end_time = ending_t
    if curr_time in timeSigsOnsets:
        idx = timeSigsOnsets.index(curr_time)
        curr_den = all_timesigs[idx].denominator
        curr_num = all_timesigs[idx].numerator
    #for tempos search between curr_time and end_time if anything occurs
    ltempos_idxs= [val for val in tempos_idxs if val >= curr_time and val <= end_time]
    if len(ltempos_idxs) == 0:
        #get the previous one
        ltempo = allBars[-1]['tempo']
    else:
        #get the last
        ltempo = tempos_vals[tempos_idxs.index(ltempos_idxs[-1])]  
        
    lBar['bar'] = len(all_downbeats)
    lBar['tempo'] = ltempo
    lBar['denominator'] = curr_den
    lBar['numerator'] = curr_num
    lBar['beat_start'] = beat_cnt
    lBar['time_start'] = curr_time
    lBar['time_end'] = end_time
    dur = (curr_num*4)/curr_den
    lBar['beat_dur'] = dur
    allBars.append(lBar)
    
            
    return allBars

def arrange_timesteps(numerator, denominator, beat_res, dur):
    '''
    @param numerator: numerator of this bar
    @param denominator: denominator of this bars
    @param beat_res: the beat resolution (global variable
    @param dur: the total durations of this bar (seconds)
    @return: the total number of timesteps of this bar according to beat_res 
            and time signature, along with the duration of each timestep in
            terms of seconds
    '''
    #calculate num timesteps for this bar
    timesteps = (4*numerator*beat_res)/denominator
    sep_ = dur/timesteps
    
    return sep_, int(timesteps)

def get_instrument_notes(anInstr, beg, end):
    '''
    @param anInstr: the PrettyMIDI input instrument
    @param beg: the beginning of this bar (seconds)
    @param end: the ending of this bar (seconds)
    @return: the adjusted PrettyMIDI instance for this bar
    
    '''
    #duration of this bar
    dur = end - beg
    #create a new pm Instrument
    newInstr = pm.Instrument(program = anInstr.program, is_drum = anInstr.is_drum,
                             name = anInstr.name)
    
    new_notes = []
    #get the notes from the instr with onsets in the range
    notes_i = deepcopy([item for item in anInstr.notes if item.start >= beg and item.start < end])
    #fix timings and offsets if needed
    if len(notes_i) > 0 :
        for i in range (0, len(notes_i)):
            notes_i[i].start = notes_i[i].start-beg
            notes_i[i].end = notes_i[i].end-beg
            if notes_i[i].end > dur:
                notes_i[i].end = dur
        new_notes.extend(notes_i)        
    #get the overallapped notes
    notes_o = deepcopy([item for item in anInstr.notes if item.start < beg and item.end > beg])
    #fix timings and onsets
    if len(notes_o) > 0 :
        for o in range (0, len(notes_o)):
            notes_o[o].start = np.float64(0.0)
            notes_o[o].end = notes_o[o].end-beg
            if notes_o[o].end > dur:
                notes_o[o].end = dur
        new_notes.extend(notes_o) 
    """
    #last case where full notes overalapping whole meters
    notes_v = deepcopy([item for item in anInstr.notes if item.start < beg and item.end > end])
    #fix timings durations
    if len(notes_v) > 0:
        for v in range(0, len(notes_v)):
            notes_v[v].start = np.float64(0.0)
            notes_v[v].end = dur
        new_notes.extend(notes_v) 
    """    
    newInstr.notes = new_notes
    
    return newInstr


def get_drum_pitch(k):
    
     if k == 'Kick':
          dru = 35   
     elif k == 'Snare':
          dru = 38
     elif k == 'Snare_Stick':
          dru = 37
     elif k == 'Closed_HH': #Closed HH 
          dru = 42
     elif k == 'Open_HH':
          dru = 46
     elif k == 'Ride':
          dru = 51
     elif k == 'Ride_Bell':
          dru = 53
     elif k == 'Tom3':
          dru = 43             
     elif k == 'Tom2':
          dru = 47  
     elif k == 'Tom1':
          dru = 50
     elif k == 'Crash1':
          dru = 49
     elif k == 'Crash2':
          dru = 57
     elif k == 'China':
          dru = 52
          
     return dru

 
                
def create_pp_instance(allEvs, onsets_ev, drums_ev):
    
    '''PYPIANOROLL EXPORT FUNCTION'''
    
    guitarPR = []
    bassPR = []
    drumsPR = []
    tempo_ids = []
    downbeats_ids = []
    
    drums_idxs = [index for index, value in enumerate(onsets_ev) if value == 'bar']
    drums_idxs.append(len(onsets_ev))

    
    for e in range(0, len(allEvs)):
        guitarPR.append(allEvs[e]['Guitar_PR'])
        bassPR.append(allEvs[e]['Bass_PR'])
        num_steps = allEvs[e]['Bass_PR'].shape[0]
        downbeats_np = np.zeros((num_steps,), dtype=bool)
        downbeats_np[0] = True
        downbeats_ids.append(downbeats_np)
        tempo_v = float(allEvs[e]['tempo'])
        tempo_np = np.zeros((num_steps,), dtype=float)
        tempo_np[:] = tempo_v 
        tempo_ids.append(tempo_np)
        drums_np = np.zeros((num_steps,128), dtype=bool)
        #for drums
        try:
            bar_prev =  drums_idxs[e]+1
            bar_new =  drums_idxs[e+1]
            onset_bar = onsets_ev[bar_prev:bar_new]
            drums_bar = drums_ev[bar_prev:bar_new]
            for o in range(0,len(onset_bar)):
                on_ts = round(float(onset_bar[o])*beat_res)
                dr_ts = drums_bar[o]
                if dr_ts != 'bar':
                    dr_pc = get_drum_pitch(dr_ts)
                    drums_np[on_ts,dr_pc] = True
             
            drumsPR.append(drums_np)
            
        except IndexError:
         
            drumsPR.append(drums_np)
                
            
        
    guitarPR = np.vstack(guitarPR)
    bassPR = np.vstack(bassPR)
    drumsPR = np.vstack(drumsPR)
    tempo_ids = np.concatenate(tempo_ids)
    downbeats_ids = np.concatenate(downbeats_ids)
    
    #create tracks
    guitarPP = pp.BinaryTrack(name='Guitar', program=26, is_drum=False, pianoroll = guitarPR)
    bassPP = pp.BinaryTrack(name='Bass', program=33, is_drum=False, pianoroll = bassPR)
    drumsPP = pp.BinaryTrack(name='Drums', program=0, is_drum=True, pianoroll = drumsPR)
    
    full_pp = pp.Multitrack(resolution = 12, tempo=tempo_ids, downbeat = downbeats_ids, tracks =[guitarPP, bassPP, drumsPP])
    
    
    return full_pp
