# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:00:03 2019

@author: John
"""
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display as ipd

import ewtpy

from scipy.io import wavfile
import pandas as pd



def wav_to_csv(input_filename):
    data, samrate = librosa.load(input_filename, sr=4000)
    plt.figure(figsize=(12, 4))
    _ = librosa.display.waveplot(data, sr=samrate)
    ipd.Audio(input_filename)
    #samrate, data = wavfile.read(input_filename)
    print('Load is Done! \n')
    print(samrate)
    print(data)
    print(type(data))

    wavData = pd.DataFrame(data)
    print(wavData)

    print('Mono .wav file\n')

    wavData.to_csv(str(input_filename[:-4] + "_resampled.csv"), index=False, index_label=None,sep=',')

    print('Save is done')

#wav_to_csv("b0078.wav")
wav_to_csv("a0007.wav")
signal = "a0007_resampled.csv"  # sig1,sig2,sig3,eeg or any other csv file with a signal
data = np.loadtxt(signal, delimiter=",")
if len(data.shape) > 1:
    f = data[:, 0]
else:
    f = data

# f = f - np.mean(f)

f /= np.max(f)

N = 2  # number of supports
detect = "locmax"  # detection mode: locmax, locmaxmin, locmaxminf
reg = 'none'  # spectrum regularization - it is smoothed with an average (or gaussian) filter
lengthFilter = 0  # length or average or gaussian filter
sigmaFilter = 0  # sigma of gaussian filter
Fs = 4000  # sampling frequency, in Hz (if unknown, set 1)

ewt, mfb, boundaries = ewtpy.EWT1D(f,
                                   N=N,
                                   log=0,
                                   detect=detect,
                                   completion=0,
                                   reg=reg,
                                   lengthFilter=lengthFilter,
                                   sigmaFilter=sigmaFilter)

print(ewt.shape)
print(ewt.shape[0])

print(boundaries)
print(mfb.shape)
print(mfb[1000,:])

# plot original signal and decomposed modes
plt.figure(figsize=(12, 4))
plt.subplot(211)
plt.plot(f)
plt.title('Original signal %s' % signal)
for x in range(ewt.shape[1]):
    plt.figure(figsize=(12, 4))
    plt.plot(ewt[:,x])
    plt.title('EWT modes %i' %x)

band = (ewt[:,2])

band_scaled = (band - band.min(axis=0)) / (band.max(axis=0) - band.min(axis=0))
print(band.min(axis=0))
print(band.max(axis=0))

print(ewt[:,2].shape)
print(band_scaled)
print(band_scaled.sum())
entropy = ((band_scaled**2)*(np.log(band_scaled**2)))
print(type(entropy))
print(np.ndarray.mean(entropy))
print(np.std(entropy))
NASE = (entropy - np.average(entropy)) / np.std(entropy)
#entropy = entropy(band)
print('Entropy: %s' %entropy)
print('NASE: %s' %NASE)

plt.figure(figsize=(12, 4))
plt.plot(NASE)
plt.title('NASE')

plt.figure(figsize=(12, 4))
plt.plot(entropy)
plt.title('Entropy')



# %% show boundaries
ff = np.fft.fft(f)
freq = 2 * np.pi * np.arange(0, len(ff)) / len(ff)

if Fs != -1:
    freq = freq * Fs / (2 * np.pi)
    boundariesPLT = boundaries * Fs / (2 * np.pi)
else:
    boundariesPLT = boundaries


ff = abs(ff[:ff.size // 2])  # one-sided magnitude
freq = freq[:freq.size // 2]
print(max(ff))

max_y = max(ff)  # Find the maximum y value
max_x = freq[ff.argmax()]  # Find the x value corresponding to the maximum y value
print(max_x, max_y)

plt.figure(figsize=(12, 4))
plt.plot(freq, ff)
for bb in boundariesPLT:
    plt.plot([bb, bb], [0, max(ff)], 'r--')
plt.title('Spectrum partitioning')
plt.xlabel('Hz')
plt.show()
