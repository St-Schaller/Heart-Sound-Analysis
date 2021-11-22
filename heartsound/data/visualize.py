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
plt.figure(figsize=(12, 4))
data, sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data, sr=sample_rate)
ipd.Audio(filename)
'''
