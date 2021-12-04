import glob

import librosa
from matplotlib import pyplot as plt

for audio in sorted(glob.iglob('/home/local/Dokumente/HeartApp/PASCAL/AB_training*.wav')):
    plt.figure(figsize=(12, 4))
    data, sample_rate = librosa.load(audio)
    _ = librosa.display.waveplot(data, sr=sample_rate)
    ipd.Audio(audio)
    plt.safefig('' + )


'''
#downsample
N, SR = librosa.load('test.wav', sr=8000) # Downsample 44.1kHz to 8kHz

resampled_signal = scipy.signal.resample( x, 8000 )

#normalize amplitude
from pydub import AudioSegment, effects  

rawsound = AudioSegment.from_file("./input.m4a", "m4a")  
normalizedsound = effects.normalize(rawsound)  
normalizedsound.export("./output.wav", format="wav")

#visualisierung

'''
