import IPython
# Plots
import matplotlib.pyplot as plt
from pylab import plot, show, figure, imshow
import mido
plt.rcParams['figure.figsize'] = (15, 6)

import numpy as np
import essentia.standard as es

def calcualte_note_differ(notes, target_notes):
    note_differ = []
    # if len(notes) <= len(target_notes):
    #     note_length = len(notes) 
    # else : 
    #     note_length = len(target_notes)

    # for i in range(note_length):
    #     note_differ.append(abs(notes[i] - target_notes[i]))
    return sum(target_notes) / len(target_notes) - sum(notes) / len(notes)


audiofile_origin = '/home/mingyu/essentiapy/koyote_original.mp3'  
audio_file_cover1 = '/home/mingyu/essentiapy/origin_snowflower_vocal.wav'  
audiofile_cover2 = '/home/mingyu/essentiapy/koyote_sixkeydown.mp3'  


loader_origin = es.EqloudLoader(filename=audiofile_origin, sampleRate=44100)
loader_cover = es.EqloudLoader(filename=audio_file_cover1, sampleRate=44100)
loader_cover2 = es.EqloudLoader(filename=audiofile_cover2, sampleRate=44100)

audio_origin = loader_origin()
audio_cover = loader_cover()
audio_cover2 = loader_cover2()

pitch_extractor = es.PredominantPitchMelodia(frameSize=2048, hopSize=128)
pitch_values_origin, pitch_confidence_origin = pitch_extractor(audio_origin)

pitch_extractor = es.PredominantPitchMelodia(frameSize=2048, hopSize=128)
pitch_values_cover, pitch_confidence_cover = pitch_extractor(audio_cover)

pitch_extractor = es.PredominantPitchMelodia(frameSize=2048, hopSize=128)
pitch_values_cover2, pitch_confidence_cover2 = pitch_extractor(audio_cover2)


# extractor = es.MFCC()
# mfccs1 = extractor(audio_origin)

# extractor = es.MFCC()
# mfccs2 = extractor(audio_cover)

# similarity_scores = []
# for mfcc1, mfcc2 in zip(mfccs1, mfccs2):
#     distance = np.linalg.norm(mfcc1 - mfcc2)
#     similarity_score = 1 / (1 + distance)
#     similarity_scores.append(similarity_score)

# 결과 출력
# print("first frame : ", similarity_scores[0])
# print("second frame : ", similarity_scores[1])

onsets, durations, notes = es.PitchContourSegmentation(hopSize=128)(pitch_values_origin, audio_origin)
onsets_cov, durations_cov, notes_cov = es.PitchContourSegmentation(hopSize=128)(pitch_values_cover, audio_cover)
onsets_cov2, durations_cov2, notes_cov2 = es.PitchContourSegmentation(hopSize=128)(pitch_values_cover2, audio_cover2)

rhythm_extractor = es.RhythmExtractor2013(method="degara")
analysis_result = rhythm_extractor(audio_origin)

bpm = analysis_result[0]
print("1. origin max notes : ",max(notes))
print("1. origin min notes : ",min(notes))
print("2. cover max notes : ",max(notes_cov))
print("2. cover min notes : ",min(notes_cov))
print("3. cover2 min notes : ",min(notes_cov2))
print("3. cover2 min notes : ",min(notes_cov2))



print("Average differ(comp with origin & cover1) : ", calcualte_note_differ(notes, notes_cov))
print("Average differ(comp with origin & cover2) : ", calcualte_note_differ(notes, notes_cov2))
print("pitch differ(comp with origin & cover2) : ", calcualte_note_differ(pitch_values_origin, pitch_values_cover))
print("pitch differ(comp with origin & cover2) : ", calcualte_note_differ(pitch_values_origin, pitch_values_cover2))
# print("BPM : ", bpm)
# PPQ = 96 # Pulses per quarter note.
# BPM = bpm # Assuming a default tempo in Ableton to build a MIDI clip.
# tempo = mido.bpm2tempo(BPM) # Microseconds per beat.

# # Compute onsets and offsets for all MIDI notes in ticks.
# # Relative tick positions start from time 0.
# offsets = onsets + durations
# silence_durations = list(onsets[1:] - offsets[:-1]) + [0]

# mid = mido.MidiFile()
# track = mido.MidiTrack()
# mid.tracks.append(track)

# for note, onset, duration, silence_duration in zip(list(notes), list(onsets), list(durations), silence_durations):
#     track.append(mido.Message('note_on', note=int(note), velocity=64,
#                               time=int(mido.second2tick(duration, PPQ, tempo))))
#     track.append(mido.Message('note_off', note=int(note),
#                               time=int(mido.second2tick(silence_duration, PPQ, tempo))))

# midi_file = '/home/mingyu/essentiapy/extracted_melody.mid'
# mid.save(midi_file)
# print("MIDI file location:", midi_file)