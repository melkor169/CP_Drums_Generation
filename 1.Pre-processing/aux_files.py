#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 16:29:23 2021

@author: bougatsas
"""

import numpy as np
import pretty_midi as pm
import music21 as m21

from copy import deepcopy
from math import ceil
from sklearn.preprocessing import OneHotEncoder
from operator import itemgetter

"""GLOBAL VARIABLES"""

beat_res = 12 #quantized 1/16 triplet


def get_reduced(pm_data, allBars_info, g_trks, b_trks, d_trks, msd_id):
    
    
    '''GUITAR'''
    if len(g_trks) > 1:
        #deepcopy the first instrument
        guitarInstr = deepcopy(pm_data.instruments[g_trks[0]])
        #start with empty notes
        guitarNotes = []   
        for bar in allBars_info:
            time_start = bar['time_start']
            time_end = bar['time_end']
            #get notes from each track
            alltrk_notes = []
            alltrk_chIDX = []
            for k in g_trks:
                notes, chord_cnt = get_bar_notes(pm_data.instruments[k], time_start, time_end)
                alltrk_notes.append(notes)
                alltrk_chIDX.append(chord_cnt)
            #get the track with the maximum chord events and or note events
            bar_notes = get_max_guitar(alltrk_notes, alltrk_chIDX)
            guitarNotes.extend(bar_notes)
            
        #set the notes to the Instr
        guitarInstr.notes = guitarNotes 
    else:
        #simply deepcopy the instrument
        guitarInstr = deepcopy(pm_data.instruments[g_trks[0]])
        

    '''BASS'''
    if len(b_trks) > 1:
        #deepcopy the first instrument
        bassInstr = deepcopy(pm_data.instruments[b_trks[0]])
        #start with empty notes
        bassNotes = []   
        for bar in allBars_info:
            time_start = bar['time_start']
            time_end = bar['time_end']
            #get notes from each track
            alltrk_notes = []
            for k in b_trks:
                notes, _ = get_bar_notes(pm_data.instruments[k], time_start, time_end)
                alltrk_notes.append(notes)
            #get the track with the maximum note events
            bar_notes = get_max_bass(alltrk_notes)
            bassNotes.extend(bar_notes)
            
        #set the notes to the Instr
        bassInstr.notes = bassNotes 
    else:
        #simply deepcopy the instrument
        bassInstr = deepcopy(pm_data.instruments[b_trks[0]])
        
    '''DRUMS'''
    #should be always one

    drumsInstr = deepcopy(pm_data.instruments[d_trks[0]]) 
    
    pm_data.instruments = [guitarInstr,bassInstr,drumsInstr]
    
    path = './reduced_out/'+msd_id+'_Reduced.mid'
    
    pm_data.write(path)
    
    return path



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


def check_phrase(phrase_bars):
    
    bass_guitar_cnt = 0
    drums_cnt = 0
    isValid = False
    
    for i in range(0,len(phrase_bars)):
        bassEV = phrase_bars[i]['Bass_EV']
        guitarEV =  phrase_bars[i]['Guitar_EV']
        drumsEV = phrase_bars[i]['Drums_EV']
        
        if bassEV or guitarEV:
            bass_guitar_cnt += 1
        if drumsEV:
            drums_cnt += 1
            
    if bass_guitar_cnt > 2 and drums_cnt > 2:
        isValid = True
        
    return isValid
        
        
def break_phrase(allBars_info,max_phrase_lgt):
    
    #define the number of breaks
    num_breaks = ceil(len(allBars_info)/max_phrase_lgt)
    #creat a dict now for new allBars_info
    allPhrases_info = {}
    bar_cnt = 0
    for br in range(0,num_breaks):
        if br != num_breaks-1:
            phrase_bars = allBars_info[bar_cnt:bar_cnt+max_phrase_lgt]
            isValid = check_phrase(phrase_bars)
            if isValid:
                phr_name = 'phrase_'+str(br)
                allPhrases_info[phr_name] = phrase_bars
            bar_cnt += max_phrase_lgt
        else: #last one
            phrase_bars = allBars_info[bar_cnt:]
            if len(phrase_bars) >= 4: #check if is smaller than 4 bars the last one
                isValid = check_phrase(phrase_bars)
                if isValid:
                    phr_name = 'phrase_'+str(br)
                    allPhrases_info[phr_name] = phrase_bars
                
    return allPhrases_info    
    

def merge_bars_fes(allBars_info, allIns_ev, allBars_pRs):
    '''
    @param allBars_info: a list with general musical info (allBars_info) for every bar 
    @param allBars_pRs: a list with each node corresponds the bar containing the PianoRolls of every track
    @return: the merged edition of these two variables
    '''
    bar_lgt = len(allBars_info)
    #fix PRs depending on length
        
    pr_diff = bar_lgt - len(allBars_pRs)
    if pr_diff !=0:
        allBars_pRs = fix_prs_lgts(allBars_pRs, pr_diff, bar_lgt, allBars_info)
        
    
    for i in range(0, len(allBars_info)):
        #set first the EV lists
        allBars_info[i]['Guitar_EV'] = allIns_ev[0][i]
        allBars_info[i]['Bass_EV'] = allIns_ev[1][i]
        allBars_info[i]['Drums_EV'] = allIns_ev[2][i]
        #add pRs
        allBars_info[i]['Guitar_PR'] = allBars_pRs[i][0]
        allBars_info[i]['Bass_PR'] = allBars_pRs[i][1]
        allBars_info[i]['Drums_PR'] = allBars_pRs[i][2]
        
    return allBars_info    
                

def get_event_based_rep(ps, allBars_info):
    #get pseudo event-based representation on non-drum tracks and event-based on drums
    s = m21.converter.parse(ps)
 
    allIns_ev = [] #Guitar, Bass
    for i in range(0,len(s.parts)-1):
        p = s.parts[i]
        if p.hasVoices():
            p = p.flattenUnnecessaryVoices(force=True) #remove (if any) voices
        #p = p.notesAndRests.stream() #keep only note events
        p = p.notes.stream() #keep only note events without rests
        all_evs = convert_pseudo_events(p, allBars_info)
        allIns_ev.append(all_evs)
        
    #call different func for drums
    p = s.parts[-1]
    if p.hasVoices():
        p = p.flattenUnnecessaryVoices(force=True) #remove (if any) voices
    p = p.notes.stream() #keep only note events without rests
    allIns_ev.append(get_drum_events(p, allBars_info))    
    
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
                                

def get_drum_events(p, allBars_info):
    #create drum events
    drum_events = [] #events per bar
    beat_cnt = 0.0 #beat counter
    for i in range(0,len(allBars_info)):
        beat_start = beat_cnt
        beat_end = allBars_info[i]['beat_dur']+beat_cnt
        #create event list for this measure
        e_bar = []     
        for e in p.getElementsByOffset(beat_start,beat_end, includeEndBoundary=False):
            #drum type of the event (offset, pitches)
            item = e.classes[0]
            #get offset
            offset = str(e.offset-beat_start)[:6]
            aEve = [offset]
            #check if it is chord or note
            
            if item == 'Chord':
                pitches = []
                for ptc in e.pitches:
                    pitches.append(ptc.midi)
                aEve.extend(pitches)
            else: #note
                pitch = e.pitch.midi
                aEve.append(pitch)
            
            e_bar.append(aEve)
        
        #for the next loop
        beat_cnt = beat_end
        #remove duplicate onsets on the drum events
        if e_bar:
            e_bar = remove_duplicates(e_bar)
        drum_events.append(e_bar)
        
    return drum_events


def remove_duplicates(e_bar):
    
    e_bar_n = []
    
    #get the first
    onset = e_bar[0][0]
    notes = e_bar[0][1:]
    
    for o in range(1, len(e_bar)):     
        onset_n = e_bar[o][0]
        if onset_n == onset:
            notes.extend(e_bar[o][1:])
        else:
            #append the previous
            aEve = [onset]
            aEve.extend(list(sorted(set(notes))))
            e_bar_n.append(aEve)
            #prepare for new
            onset = e_bar[o][0]
            notes = e_bar[o][1:]
            
    #append the last one
    aEve = [onset]
    aEve.extend(list(sorted(set(notes))))
    e_bar_n.append(aEve)    
    
    return e_bar_n                
            
            





def get_max_guitar(alltrk_notes, alltrk_chIDX):
    
    note_lgts = []
    for n in alltrk_notes:
        note_lgts.append(len(n))
    
    #prefer tracks with chord events
    #check if there is chord track 
    if max(alltrk_chIDX) >= 2:
        g_trk = alltrk_chIDX.index(max(alltrk_chIDX))
    else:
        #rely on the number of events
        g_trk = note_lgts.index(max(note_lgts))
        
    bar_notes = alltrk_notes[g_trk]
    
    return bar_notes


def get_max_bass(alltrk_notes):
    
    note_lgts = []
    for n in alltrk_notes:
        note_lgts.append(len(n))
    
    b_trk = note_lgts.index(max(note_lgts))
        
    bar_notes = alltrk_notes[b_trk]
    
    return bar_notes
        


def get_bar_notes(pm_instr, time_start, time_end):
    
    notes = []
    notes_onsets = []
    
    for note in pm_instr.notes:
        if note.start >= time_start and note.start < time_end:
            notes.append(note)
            notes_onsets.append(note.start)
            
    #detect if there are chord events. The duration should be not less than thresh
    thresh = (time_end-time_start)/24
    #chord counter note
    cnt = 0
  
    for n in range(0,len(notes_onsets)-1):
        right_b = notes_onsets[n]
        left_b = notes_onsets[n+1]
        if left_b - right_b < thresh:
            cnt += 1
    
            
    return notes, cnt
    


def calc_tension_range(aDiam_seq, aCent_seq , aKey_seq):
    
    aDiamR = [min(aDiam_seq), max(aDiam_seq)]
    aCentR = [min(aCent_seq), max(aCent_seq)]
    aKeyR = [min(aKey_seq), max(aKey_seq)] 
    
    return aDiamR, aCentR, aKeyR


def check_pm_instrs(pm_data):
    
    is_ok = False
    guitar_check = False
    bass_check = False
    drums_check = False
    #check if drums, bass and guitar tracks exist
    for instr in pm_data.instruments:
        if 'Guitar' in instr.name:
            guitar_check = True
        if 'Bass' in instr.name:
            bass_check = True
        if 'Drums'in instr.name:
            drums_check = True
            
    if guitar_check and bass_check and drums_check:
        is_ok = True
    
    return is_ok


def merge_pm(pm_data):
    
    newInstrs = []

    newInstrs.append(pm_data.instruments[0])
    prevName = newInstrs[-1].name
    if 'Drums' in prevName:
        prevName = 'Drums'
    for i in range(1,len(pm_data.instruments)):
        newInstr = pm_data.instruments[i]
        newName = pm_data.instruments[i].name
        if 'Drums' in newName:
            newName = 'Drums'
        if newName != prevName:
            #add it to the list
            newInstrs.append(pm_data.instruments[i])
            prevName = newInstrs[-1].name
        else: #it is the same
            #get all notes and append them in the previous instrument
            newInstrs[-1].notes.extend(newInstr.notes)
            
            
    #get if drums, bass and guitar tracks indexes
    g_trks = []
    b_trks = []
    d_trks = []
    for i in range(0, len(newInstrs)):
        instr = newInstrs[i]
        if 'Guitar' in instr.name:
            g_trks.append(i)
        if 'Bass' in instr.name:
            b_trks.append(i)
        if 'Drums'in instr.name:
            d_trks.append(i)
            
    pm_data.instruments = newInstrs
    
    return pm_data, g_trks, b_trks, d_trks


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
            #and the overalapping notes index
            newInstr, overlap_idx = get_instrument_notes(anInstr, beg, end)
            #then create a quantized PR
            inst_pp = get_pianoRoll(timestep_dur, total_timesteps, dur, newInstr, overlap_idx)
            aBar_pp.append(inst_pp)
            
        allBars_pRs.append(aBar_pp)
        
    return allBars_info, allBars_pRs


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
    
    return newInstr, len(notes_i)


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

def get_pianoRoll(timestep_dur, total_timesteps, dur, anInstrument, overlap_idx):
    '''
    @param timestep_dur: the duration of each timestep
    @param total_timesteps: the total timesteps
    @param dur: the overall duration of this bar
    @anInstrument: the input instrument containing notes for this bar in PrettyMIDI instance
    @return: a quantized (to the next 1/16th) pianoroll for this bar
    '''
    
    #create a PianoRoll instance
    aPR =  np.zeros((total_timesteps, 128), dtype=int)
    
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
    
    for n in range(0, len(anInstrument.notes)):
        note = anInstrument.notes[n]
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
                    #check if it is non overallping
                    if n < overlap_idx:
                        aPR[ons_idx ,note.pitch] = note.velocity                 
                        aPR[ons_idx+1:off_idx+1 ,note.pitch] = -note.velocity 
                    else:
                        aPR[ons_idx:off_idx+1 ,note.pitch] = -note.velocity 
            else:
                #assign only onset (with the velocity)
                aPR[ons_idx ,note.pitch] = note.velocity
                    
        except StopIteration:
            #rejected note
            continue
                
                
    return aPR.astype(bool)


def create_onehot_dict(all_occs):
    
    onehotEnc = OneHotEncoder(handle_unknown='ignore')
    onehotEnc.fit(np.array(all_occs).reshape(-1,1))
    
    return onehotEnc                
        

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


def get_groupID(i, numOfBars):
    
    if i == 0:
        groupID = 'start1'
    elif i == 1:
        groupID = 'start2'
    elif i == numOfBars-2:
        groupID = 'end1'
    elif i == numOfBars-1:
        groupID = 'end2'
    else:
        groupID = 'middle'
        
    return groupID


def create_enc_dec(allBars_info):
    
    numOfBars = len(allBars_info)
    
    Enc_Onset = [] #parallel streams
    Enc_Instr = []
    Enc_Type = []
    Enc_Duration = [] 
    Enc_Value = []
    
    Dec_Onsets = [] #decoder stream for onsets only
    Dec_Drums = [] #decoder stream for drums elements
    
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
        
        '''Decoder stage'''
        anDec_onsets = ['bar'] 
        anDec_drums = ['bar']
 
        drums = allBars_info[i]['Drums_EV']
        drumsEV = []
        if drums: #not empty
            for d in drums: #convert Drum Pitches to events
                aDruEV = [d[0]] #onset
                #get a row of events
                events = d[1:] 
                for el in events:
                    dru = get_drum_element(el)
                    if dru != 'NotFound':
                        aDruEV.append(dru)
                #Checkin the event        
                #1.Remove duplicates and sort the elements
                aDruEV = list(sorted(set(aDruEV)))
                #2.remove irregular events
                aDruEV = rm_drum_ev(aDruEV)
                if len(aDruEV) > 1: #2.not empty events
                   drumsEV.append(aDruEV)
            #separate them
            for d in drumsEV:
                evs_onset = d[0]
                evs = d[1:]
                for e in range(0, len(evs)):
                    anDec_onsets.append(evs_onset)
                    anDec_drums.append(evs[e])
            
        #extend the Decoder streams
        Dec_Onsets.extend(anDec_onsets)
        Dec_Drums.extend(anDec_drums)
        
    return Enc_Onset, Enc_Instr, Enc_Type, Enc_Duration, Enc_Value, Dec_Onsets, Dec_Drums   

                        

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