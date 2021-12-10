import glob
import os
import signal
from math import pi, log, log2
from pathlib import Path

import dit
import librosa
import librosa.display
import numpy as np
import pandas as pd
import scipy
from librosa import lpc
from numpy import mean, diff, std, var, double, trapz
from numpy.fft import fft
from pywt import wavedec
from pyyawt import appcoef, detcoef
from scipy.stats import moment, skew, kurtosis
import nnresample
from dit.other import renyi_entropy, tsallis_entropy
import pyyawt




def extract_features_without_segmentation(audiodata):

    audio, sample_rate = librosa.load(audiodata, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccsscaled = np.mean(mfccs.T ,axis=0)

    Fs = 2000
    f1 = np.mean(np.min(mfccs, axis=1)) # 1

    m4 = moment((np.max(mfccs, axis=1)), 4)
    f5 = abs(np.power(m4, (1 / 4))) # 5
    m4 = moment((skew(mfccs)), 4)
    f6 = abs(np.power(m4, (1 / 4))) # 6

    # --------------------------------------------------------------------------
    level = 5
    a5, det5, det4, det3, det2, det1 = wavedec(data=audio, wavelet='db4', level=level)

    f7 = log2(var(det3)) # 7
    try:
        xx,_ = librosa.load(audiodata, sr=16)
        Pfb1, Pfb2, Pfb3, Pfb4, Pfb5, Pfb6, Pfb7, Pfb8, Pfb9, Pfb10 = Spectral(xx)  # 13 14
    except Exception:
        Pfb1 = Pfb2 = Pfb3 = Pfb4 = Pfb5 = Pfb6 = Pfb7 = Pfb8 = Pfb9 = Pfb10 = 0

    lp = lpc(double(audio), 10)
    lp2 = lp[1]
    lp4 = lp[3]
    lp7 = lp[6]
    lp9 = lp[8]
    lp10 = lp[9]
    lp11 = lp[10]

    spectral_centroid = librosa.feature.spectral_centroid(audio, sr=sample_rate)  #26 NICHT MPSD Centroid sondern Spectral Centroid

    _, counts = np.unique(audio, return_counts=True)

    shannon = scipy.stats.entropy(counts) #29


    #tsallis = tsallis_entropy(counts,  2) #31

    _, counts = np.unique(a5, return_counts=True)
    shannon_a5 = scipy.stats.entropy(counts) # 32

    _, counts = np.unique(det5, return_counts=True)
    #renyi_d5 = renyi_entropy(counts, 2)# 36

    _, counts = np.unique(det4, return_counts=True)
    shannon_d4 = scipy.stats.entropy(counts)# 38

    '''
    print(f"MFCCs scaled: {mfccsscaled} \n"
          f"F1: {f1} \n"
          f"F5: {f5} \n"
          f"F6: {f6} \n"
          f"F7: {f7} \n"
          f"Pfb1: {Pfb1} \n"
          f"Pfb2: {Pfb2} \n "
          f"Pfb3: {Pfb3} \n"
          f"Pfb4: {Pfb4} \n"
          f"Pfb5: {Pfb5} \n"
          f"Pfb6: {Pfb6}\n"
          f"Pfb7: {Pfb7} \n"
          f"Pfb8: {Pfb8} \n"
          f"Pfb9: {Pfb9} \n"
          f"Pfb10: {Pfb10} \n"
          f"lp2: {lp2} \n"
          f"Lp4:  {lp4} \n"
          f"lp7; {lp7} \n"
          f"lp9: {lp9} \n"
          f"lp10: {lp10} \n"
          f"lp11: {lp11} \n"
          f"spectral centroid: {spectral_centroid} \n"
          f"shannon: {shannon} \n"
          f"shannon_a5: {shannon_a5} \n"
          f"shannon_d4: {shannon_d4}")
          
          '''

    # Missing LPC; Tsalllis, Reny
    return mfccsscaled, spectral_centroid, f1, f5, f6, f7, Pfb1, Pfb2, Pfb3, Pfb4, Pfb5, Pfb6, Pfb7, Pfb8, Pfb9, Pfb10, lp2, lp4, lp7, lp9, lp10, lp11,  shannon, shannon_a5, shannon_d4

def extract_features_with_segmentation(PCG,states):
    #Feature calculation
    try:
        m_RR = round(mean(diff(states.iloc[:, 0]))) # mean value of RR intervals
        sd_RR = round(std(diff(states.iloc[:, 0]))) # standard deviation(SD) value of RR intervals
        mean_IntS1 = round(mean(states.iloc[:, 1]-states.iloc[:, 0])) # mean value of S1 intervals
        sd_IntS1 = round(std(states.iloc[:, 1]-states.iloc[:, 0])) # SD value of S1 intervals
        mean_IntS2 = round(mean(states.iloc[:, 3]-states.iloc[:, 2])) # mean value of S2 intervals
        sd_IntS2 = round(std(states.iloc[:, 3]-states.iloc[:, 2])) # SD value of S2 intervals
        mean_IntSys = round(mean(states.iloc[:, 2]-states.iloc[:, 1])) # mean value of systole intervals
        sd_IntSys = round(std(states.iloc[:, 2]-states.iloc[:, 1])); # SD value of systole intervals
        mean_IntDia = round(mean(states.iloc[1:, 0]-states.iloc[0:-1, 3]))  # mean value of diastole intervals
        sd_IntDia = round(std(states.iloc[1:, 0]-states.iloc[0:-1, 3])) # SD value of diastole intervals
    except Exception:
        m_RR = sd_RR = mean_IntS1 = sd_IntS1 =  mean_IntS2 = sd_IntS2 = mean_IntSys = sd_IntSys = mean_IntDia = sd_IntDia = 0

    length = len(states)
    R_SysRR = R_DiaRR = R_SysDia = P_S1 = P_Sys = P_S2 = P_Dia = P_SysS1 = P_DiaS2 = SK_Sys = SK_Dia = SK_S1 = SK_S2 = KU_S1 = KU_Sys = KU_Dia = KU_S2 = [0] * length

    for i in range(length - 1):
        R_SysRR[i] = (states.iloc[i, 2]-states.iloc[i, 1]) /  (states.iloc[i + 1, 0]-states.iloc[i, 0]) * 100
        R_DiaRR[i] =  (states.iloc[i + 1, 0]-states.iloc[i, 3]) /  (states.iloc[i + 1, 0]-states.iloc[i, 0]) * 100
        R_SysDia[i] = R_SysRR[i] / R_DiaRR[i] * 100

        P_S1[i] = sum(abs(PCG[states.iloc[i, 0]:states.iloc[i, 1]])) / (states.iloc[i, 1]-states.iloc[i, 0])
        P_Sys[i] = sum(abs(PCG[states.iloc[i, 1]:states.iloc[i, 2]]))  / (states.iloc[i, 2]-states.iloc[i, 1])
        P_S2[i] = sum(abs(PCG[states.iloc[i , 2]:states.iloc[i, 3]]))  / (states.iloc[i , 3]-states.iloc[i, 2])
        P_Dia[i] = sum(abs(PCG[states.iloc[i, 3]:states.iloc[i+1,0]]))  / (states.iloc[i +1 , 0]-states.iloc[i, 3])

        if P_S1[i] > 0:
            P_SysS1[i] = P_Sys[i] / P_S1[i] * 100
        else:
            P_SysS1[i] = 0
        if P_S2[i] > 0:
            P_DiaS2[i] = P_Dia[i] / P_S2[i] * 100
        else:
            P_DiaS2[i] = 0

        SK_S1[i] = skew(PCG[states.iloc[i, 0]:states.iloc[i, 1]])
        SK_Sys[i] = skew(PCG[states.iloc[i, 1]:states.iloc[i, 2]])
        SK_S2[i] = skew(PCG[states.iloc[i, 2]:states.iloc[i, 3]])
        SK_Dia[i] = skew(PCG[states.iloc[i, 3]:states.iloc[i + 1,0]])

        KU_S1[i] = kurtosis(PCG[states.iloc[i,0]:states.iloc[i,1]])
        KU_Sys[i] = kurtosis(PCG[states.iloc[i,1]:states.iloc[i, 2]])
        KU_S2[i] = kurtosis(PCG[states.iloc[i, 2]:states.iloc[i, 3]])
        KU_Dia[i] = kurtosis(PCG[states.iloc[i, 3]:states.iloc[i + 1,0]])


    m_Ratio_SysRR = mean(R_SysRR) # mean value of the interval ratios between systole and RR in each heart beat
    sd_Ratio_SysRR = std(R_SysRR) # SD value of the interval ratios between systole and RR in each heart beat
    m_Ratio_DiaRR = mean(R_DiaRR) # mean value of the interval ratios between diastole and RR in each heart beat
    sd_Ratio_DiaRR = std(R_DiaRR) # SD value of the interval ratios between diastole and RR in each heart beat
    m_Ratio_SysDia = mean(R_SysDia) # mean value of the interval ratios between systole and diastole in each heart beat
    sd_Ratio_SysDia = std(R_SysDia) # SD value of the interval ratios between systole and diastole in each heart beat

    values_sys = [x for x in P_SysS1 if (x > 0) & (x < 100)]
    if len(values_sys) > 1:
        m_Amp_SysS1 = mean(values_sys) # mean value of the mean absolute amplitude ratios between systole period and S1 period in each heart beat
        sd_Amp_SysS1 = std(values_sys) # SD value of the mean absolute amplitude ratios between systole period and S1 period in each heart beat
    else:
        m_Amp_SysS1 = 0
        sd_Amp_SysS1 = 0
    values_dia = [x for x in P_DiaS2 if (x > 0) & (x < 100)]
    if len(values_dia) > 1:
        m_Amp_DiaS2 = mean(values_dia) # mean value of the mean absolute amplitude ratios between diastole period and S2 period in each heart beat
        sd_Amp_DiaS2 = std(values_dia) # SD value of the mean absolute amplitude ratios between diastole period and S2 period in each heart beat
    else:
        m_Amp_DiaS2 = 0
        sd_Amp_DiaS2 = 0

    mSK_S1 = mean(SK_S1)
    sdSK_S1 = std(SK_S1)
    mSK_Sys = mean(SK_Sys)
    sdSK_Sys = std(SK_Sys)
    mSK_S2 = mean(SK_S2)
    sdSK_S2 = std(SK_S2)
    mSK_Dia = mean(SK_Dia)
    sdSK_Dia = std(SK_Dia)

    mKU_S1 = mean(KU_S1)
    sdKU_S1 = std(KU_S1)
    mKU_Sys = mean(KU_Sys)
    sdKU_Sys = std(KU_Sys)
    mKU_S2 = mean(KU_S2)
    sdKU_S2 = std(KU_S2)
    mKU_Dia = mean(KU_Dia)
    sdKU_Dia = std(KU_Dia)

    return m_RR, sd_RR,  mean_IntS1, sd_IntS1,  mean_IntS2, sd_IntS2,  mean_IntSys, sd_IntSys,  mean_IntDia,\
                             sd_IntDia, m_Ratio_SysRR, sd_Ratio_SysRR, m_Ratio_DiaRR, sd_Ratio_DiaRR, m_Ratio_SysDia,\
                             sd_Ratio_SysDia, m_Amp_SysS1, sd_Amp_SysS1, m_Amp_DiaS2, sd_Amp_DiaS2, mSK_S1 , sdSK_S1\
                            ,mSK_Sys, sdSK_Sys, mSK_S2, sdSK_S2, mSK_Dia, sdSK_Dia, mKU_S1, sdKU_S1,\
                             mKU_Sys, sdKU_Sys, mKU_S2, sdKU_S2, mKU_Dia, sdKU_Dia


def Spectral(audiodata):
    fb1 = [0, 0.1]
    fb2 = [0.1, 0.2]
    fb3 = [0.2, 0.3]
    fb4 = [0.3, 0.4]
    fb5 = [0.4, 0.5]
    fb6 = [0.5, 0.6]
    fb7 = [0.6, 0.7]
    fb8 = [0.7, 0.8]
    fb9 = [0.8, 0.9]
    fb10 = [0.9, 1]

    #F, PSD = signal.welch(audiodata, fs=2000, window=signal.windows.hamming(len(window)), nfft=num_fft, scaling='density', return_onesided=True, detrend=False) # uses a hamming window
    F, PSD = scipy.signal.welch(audiodata)


    #-----------  find the indexes corresponding bands - -----------------
    ifb1 = (F >= fb1[0]) & (F <= fb1[1])
    ifb2 = (F >= fb2[0]) & (F <= fb2[1])
    ifb3 = (F >= fb3[0]) & (F <= fb3[1])
    ifb4 = (F >= fb4[0]) & (F <= fb4[1])
    ifb5 = (F >= fb5[0]) & (F <= fb5[1])
    ifb6 = (F >= fb6[0]) & (F <= fb6[1])
    ifb7 = (F >= fb7[0]) & (F <= fb7[1])

    ifb8 = (F >= fb8[0]) & (F <= fb8[1])
    ifb9 = (F >= fb9[0]) & (F <= fb9[1])
    ifb10 = (F >= fb10[0]) & (F <= fb10[1])
    # ------------------  calcute areas, within the freq bands(ms ^ 2) - -----------------
    Ifb1 = trapz(F[ifb1], PSD[ifb1])
    Ifb2 = trapz(F[ifb2], PSD[ifb2])
    Ifb3 = trapz(F[ifb3], PSD[ifb3])

    Ifb4 = trapz(F[ifb4], PSD[ifb4])
    Ifb5 = trapz(F[ifb5], PSD[ifb5])
    Ifb6 = trapz(F[ifb6], PSD[ifb6])
    Ifb7 = trapz(F[ifb7], PSD[ifb7])

    Ifb8 = trapz(F[ifb8], PSD[ifb8])
    Ifb9 = trapz(F[ifb9], PSD[ifb9])
    Ifb10 = trapz(F[ifb10], PSD[ifb10])

    aTotal = Ifb1 + Ifb2 + Ifb3 + Ifb4 + Ifb5 + Ifb6 + Ifb7 + Ifb8 + Ifb9 + Ifb10
    #------------------ calculate areas relative to the total area( %) ------------------
    Pfb1 = (Ifb1 / aTotal) * 100
    Pfb2 = (Ifb2 / aTotal) * 100
    Pfb3 = (Ifb3 / aTotal) * 100
    Pfb4 = (Ifb4 / aTotal) * 100
    Pfb5 = (Ifb5 / aTotal) * 100
    Pfb6 = (Ifb6 / aTotal) * 100
    Pfb7 = (Ifb7 / aTotal) * 100
    Pfb8 = (Ifb8 / aTotal) * 100
    Pfb9 = (Ifb9 / aTotal) * 100
    Pfb10 = (Ifb10 / aTotal) * 100

    return Pfb1, Pfb2, Pfb3, Pfb4, Pfb5, Pfb6, Pfb7, Pfb8, Pfb9, Pfb10

def run_feature_extraction(path = '/home/local/Dokumente/HeartApp/All_Data_splitted/All_Data_splitted_training'):
    feature_data = []
    filenames = []
    colnames = ['filename', 'labels']
    labels = pd.read_csv('/home/local/Dokumente/HeartApp/labels_all_data_splitted_combined.csv', names=colnames,
                         header=None)
    print(labels)
    column_names = ['mfccsscaled', 'lpc', 'f1', 'f5', 'f6', 'f7', 'f14', 'f15', 'f161821232425', 'spectral_centroid',
                    'shannon', 'tsallis', 'shannon_a5', 'renyi_d5', 'shannon_d4', 'm_RR', 'sd_RR',  'mean_IntS1', 'sd_IntS1',  'mean_IntS2',
                    'sd_IntS2',  'mean_IntSys', 'sd_IntSys',  'mean_IntDia', 'sd_IntDia', 'm_Ratio_SysRR', 'sd_Ratio_SysRR',
                    'm_Ratio_DiaRR', 'sd_Ratio_DiaRR', 'm_Ratio_SysDia', 'sd_Ratio_SysDia', 'm_Amp_SysS1', 'sd_Amp_SysS1',
                    'm_Amp_DiaS2', 'sd_Amp_DiaS2', 'mSK_S1' , 'sdSK_S1' ,'mSK_Sys', 'sdSK_Sys', 'mSK_S2', 'sdSK_S2', 'mSK_Dia',
                    'sdSK_Dia', 'mKU_S1', 'sdKU_S1', 'mKU_Sys', 'sdKU_Sys', 'mKU_S2', 'sdKU_S2', 'mKU_Dia', 'sdKU_Dia']
    for audiopath in sorted(glob.iglob(path + '/*.wav')):
        filename = Path(audiopath).stem
        filenames.append(filename)
        print(f"Filename: {filename}")

        features = extract_features_without_segmentation(audiopath)
        statepath = '/home/local/Dokumente/HeartApp/Segmentation_annotations/All_Data_splitted/All_Data_splitted_training/' + filename + '.csv'
        if (os.path.exists(statepath)):
            states = pd.read_csv(statepath, header=None)
            audio, sample_rate = librosa.load(audiopath)
            segmentation_features = (extract_features_with_segmentation(audio, states))
            features = list(features + segmentation_features)
        feature_data.append(features)


    feature_data = pd.DataFrame(feature_data)
    feature_data.insert(0, 'filename', filenames)
    print(feature_data)
    df_merged = pd.merge(feature_data, labels, how='inner', on='filename')
    print(df_merged)
    path = '/home/local/Dokumente/HeartApp/Feature_Extraction/'
    df_merged.to_csv(path + 'Manual_Feature_Extraction_Training_Splitted.csv', index=False)

def convert_list_values_to_columns(inputtrain, inputtest, outputfolder):
    df_train = pd.read_csv(inputtrain)
    df_test = pd.read_csv(inputtest)
    df_train = pd.concat([df_train, df_test], ignore_index=True)
    filenames = df_train['filename']
    mfcc_columns = df_train['0']
    mfcc_columns = mfcc_columns.str.replace(r'[][]', '', regex=True)
    mfcc_columns = mfcc_columns.str.split(expand=True)
    spectral_centroids = df_train['1']
    spectral_centroids = spectral_centroids.str.replace(r'[][]', '', regex=True)
    spectral_centroids = spectral_centroids.str.split(expand=True)
    print(spectral_centroids)
    rest_data = df_train.iloc[:, 3:]
    print(rest_data)
    all_data = filenames.to_frame().join(mfcc_columns)
    print(all_data)
    all_data = pd.concat([all_data, spectral_centroids, rest_data], axis=1, ignore_index=True)
    print(all_data)
    path = outputfolder
    all_data.to_csv(path + 'name', index=False)

