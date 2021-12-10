import glob
from pathlib import Path

import librosa
from matplotlib import pyplot as plt
import librosa.display

def visualize_waveform(inputfolder, outputfolder):
    for audio in sorted(glob.iglob(inputfolder + '/*.wav')):
        plt.figure(figsize=(12, 4))
        data, sample_rate = librosa.load(audio)
        librosa.display.waveplot(data, sr=sample_rate, x_axis='time')
        filename = Path(audio).stem
        plt.title(filename)
        plt.savefig(outputfolder + filename)




