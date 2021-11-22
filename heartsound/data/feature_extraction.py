import signal
from math import pi, log, log2

import librosa
import librosa.display
import numpy as np
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




def extract_features_without_segmentation(audiodata,hamming):

    try:
        audio, sample_rate = librosa.load(audiodata, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T ,axis=0)

        #Zabihi features
        Fs = 2000

        MFCCs = librosa.feature.mfcc(audiodata, Fs, 0.025 * Fs, 0.010 * Fs, 0.97, hamming, [1, 1000], 26, 14, 22)[0]

        f1 = mean(min(MFCCs)) # 1
        m4 = moment((max(MFCCs)), 4, 2)
        f5 = abs(np.power(m4, (1 / 4))) # 5
        m4 = moment((skew(MFCCs)), 4, 2)
        f6 = abs(np.power(m4, (1 / 4))) # 6
        # --------------------------------------------------------------------------
        level = 5
        [c, l] = wavedec(audiodata, level, 'db4')

        a5 = appcoef(c, l, 'db4')
        det5 = detcoef(c, l, level)
        det4 = detcoef(c, l, level - 1)
        det3 = detcoef(c, l, level - 2)

        f7 = log2(var(det3)) # 7

        xx = nnresample.resample(audiodata, 16)
        f14 ,_ ,f15 = Spectral(xx) # 13 14

        lp, g = lpc(double(audiodata), 10)
        f161821232425 = lp([2, 4, 7, 9, 10, 11]) # 16 18 21 23 24 25

        spectral_centroid = librosa.feature.spectral_centroid(audiodata, sr=sample_rate)  #26 NICHT MPSD Centroid sondern Spectral Centroid

        counts = audiodata.value_counts()
        shannon = scipy.stats.entropy(counts) #29
        tsallis = tsallis_entropy(counts) #31

        counts = a5.value_counts()
        shannon_a5 = scipy.stats.entropy(counts) # 32

        counts = det5.value_counts()
        renyi_d5 = renyi_entropy(counts)# 36

        counts = det4.value.counts()
        shannon_d4 = scipy.stats.entropy(counts)# 38

    except Exception as e:
        print("Error encountered while parsing file: ", audiodata)
        return None

    return mfccsscaled, f1, f5, f6, f7, f14, f15, f161821232425, spectral_centroid, shannon, tsallis, shannon_a5, renyi_d5, shannon_d4

def extract_features_with_segmentation(PCG,states):
    #Feature calculation
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

    length = len(states) - 1
    R_SysRR = R_DiaRR = R_SysDia = P_S1 = P_Sys = P_S2 = P_Dia = P_SysS1 = P_DiaS2 = SK_Sys = SK_Dia = SK_S1 = SK_S2 = KU_S1 = KU_Sys = KU_Dia = KU_S2 = [0] * length

    for i in range(len(states)):
        R_SysRR[i] = (states.iloc[i, 2]-states.iloc[i, 1]) /  (states.iloc[i + 1, 0]-states.iloc[i, 0]) * 100
        R_DiaRR[i] =  (states.iloc[i + 1, 0]-states.iloc[i, 3]) /  (states.iloc[i + 1, 0]-states.iloc[i, 0]) * 100
        R_SysDia[i] = R_SysRR(i) / R_DiaRR(i) * 100
        print(type(PCG))
        P_S1[i] = sum(abs(PCG(states[i][0],states[i][1]))) / (states.iloc[i, 1]-states.iloc[i, 0])
        P_Sys[i] = sum(abs(PCG(states[i][1],states[i][2])))  / (states.iloc[i, 2]-states.iloc[i, 1])
        P_S2[i] = sum(abs(PCG(states[i][2],states[i][3])))  / (states.iloc[i , 3]-states.iloc[i, 2])
        P_Dia[i] = sum(abs(PCG(states[i][3],states[i + 1][1])))  / (states.iloc[i + 1 , 0]-states.iloc[i, 3])
        #P_S1[i] = sum(abs(PCG(A(i, 1):A(i, 2)))) / (A(i, 2) - A(i, 1))
        #P_Sys[i] = sum(abs(PCG(A(i, 2):A(i, 3)))) / (A(i, 3) - A(i, 2))
        #P_S2[i] = sum(abs(PCG(A(i, 3):A(i, 4)))) / (A(i, 4) - A(i, 3))
        #P_Dia[i] = sum(abs(PCG(A(i, 4):A(i + 1, 1)))) / (A(i + 1, 1) - A(i, 4))
        if P_S1[i] > 0:
            P_SysS1[i] = P_Sys[i] / P_S1[i] * 100
        else:
            P_SysS1[i] = 0
        if P_S2[i] > 0:
            P_DiaS2[i] = P_Dia[i] / P_S2[i] * 100
        else:
            P_DiaS2[i] = 0

        SK_S1[i] = skew(PCG(states[i][0],states[i][1]))
        SK_Sys[i] = skew(PCG(states[i][1],states[i][2]))
        SK_S2[i] = skew(PCG(states[i][2],states[i][3]))
        SK_Dia[i] = skew(PCG(states[i][3],states[i + 1][0]))

        KU_S1[i] = kurtosis(PCG(states[i][0],states[i][1]))
        KU_Sys[i] = kurtosis(PCG(states[i][1],states[i][2]))
        KU_S2[i] = kurtosis(PCG(states[i][2],states[i][3]))
        KU_Dia[i] = kurtosis(PCG(states[i][3],states[i + 1][0]))


    m_Ratio_SysRR = mean(R_SysRR) # mean value of the interval ratios between systole and RR in each heart beat
    sd_Ratio_SysRR = std(R_SysRR) # SD value of the interval ratios between systole and RR in each heart beat
    m_Ratio_DiaRR = mean(R_DiaRR) # mean value of the interval ratios between diastole and RR in each heart beat
    sd_Ratio_DiaRR = std(R_DiaRR) # SD value of the interval ratios between diastole and RR in each heart beat
    m_Ratio_SysDia = mean(R_SysDia) # mean value of the interval ratios between systole and diastole in each heart beat
    sd_Ratio_SysDia = std(R_SysDia) # SD value of the interval ratios between systole and diastole in each heart beat

    indx_sys = np.argwhere(P_SysS1 > 0 & P_SysS1 < 100) # avoid the flat line signal
    if len(indx_sys) > 1:
        m_Amp_SysS1 = mean(P_SysS1(indx_sys)) # mean value of the mean absolute amplitude ratios between systole period and S1 period in each heart beat
        sd_Amp_SysS1 = std(P_SysS1(indx_sys)) # SD value of the mean absolute amplitude ratios between systole period and S1 period in each heart beat
    else:
        m_Amp_SysS1 = 0
        sd_Amp_SysS1 = 0
    indx_dia = np.argwhere(P_DiaS2 > 0 & P_DiaS2 < 100)
    if len(indx_dia) > 1:
        m_Amp_DiaS2 = mean(P_DiaS2(indx_dia)) # mean value of the mean absolute amplitude ratios between diastole period and S2 period in each heart beat
        sd_Amp_DiaS2 = std(P_DiaS2(indx_dia)) # SD value of the mean absolute amplitude ratios between diastole period and S2 period in each heart beat
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



    return m_RR, sd_RR,  mean_IntS1, sd_IntS1,  mean_IntS2, sd_IntS2,  mean_IntSys, sd_IntSys,  mean_IntDia, sd_IntDia, m_Ratio_SysRR, sd_Ratio_SysRR, m_Ratio_DiaRR, sd_Ratio_DiaRR, m_Ratio_SysDia, sd_Ratio_SysDia, m_Amp_SysS1, sd_Amp_SysS1, m_Amp_DiaS2, sd_Amp_DiaS2



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
    F, PSD = signal.welch(audiodata)

    #-----------  find the indexes corresponding bands - -----------------
    ifb1 = (F >= fb1(1)) & (F <= fb1(2))
    ifb2 = (F >= fb2(1)) & (F <= fb2(2))
    ifb3 = (F >= fb3(1)) & (F <= fb3(2))
    ifb4 = (F >= fb4(1)) & (F <= fb4(2))
    ifb5 = (F >= fb5(1)) & (F <= fb5(2))
    ifb6 = (F >= fb6(1)) & (F <= fb6(2))
    ifb7 = (F >= fb7(1)) & (F <= fb7(2))

    ifb8 = (F >= fb8(1)) & (F <= fb8(2))
    ifb9 = (F >= fb9(1)) & (F <= fb9(2))
    ifb10 = (F >= fb10(1)) & (F <= fb10(2))
    # ------------------  calcute areas, within the freq bands(ms ^ 2) - -----------------
    Ifb1 = trapz(F(ifb1), PSD(ifb1))
    Ifb2 = trapz(F(ifb2), PSD(ifb2))
    Ifb3 = trapz(F(ifb3), PSD(ifb3))
    Ifb4 = trapz(F(ifb4), PSD(ifb4))
    Ifb5 = trapz(F(ifb5), PSD(ifb5))
    Ifb6 = trapz(F(ifb6), PSD(ifb6))
    Ifb7 = trapz(F(ifb7), PSD(ifb7))

    Ifb8 = trapz(F(ifb8), PSD(ifb8))
    Ifb9 = trapz(F(ifb9), PSD(ifb9))
    Ifb10 = trapz(F(ifb10), PSD(ifb10))

    aTotal = Ifb1 + Ifb2 + Ifb3 + Ifb4 + Ifb5 + Ifb6 + Ifb7 + Ifb8 + Ifb9 + Ifb10
    #------------------ calculate areas relative to the total area( %) ------------------
    # Pfb1 = (Ifb1 / aTotal) * 100
    # Pfb2 = (Ifb2 / aTotal) * 100
    # Pfb3 = (Ifb3 / aTotal) * 100
    # Pfb4 = (Ifb4 / aTotal) * 100
    # Pfb5 = (Ifb5 / aTotal) * 100
    # Pfb6 = (Ifb6 / aTotal) * 100
    # Pfb7 = (Ifb7 / aTotal) * 100
    Pfb8 = (Ifb8 / aTotal) * 100
    Pfb9 = (Ifb9 / aTotal) * 100
    Pfb10 = (Ifb10 / aTotal) * 100

    return Pfb8, Pfb9, Pfb10


'''
potes
function features = get_features_frequency(PCG,idx_states)
    NFFT = 256;
    f = (0:NFFT/2-1)/(NFFT/2)*500;
    freq_range = [25,45;45,65;65,85;85,105;105,125;125,150;150,200;200,300;300,500];
    p_S1  = nan(size(idx_states,1)-1,NFFT/2);
    p_Sys = nan(size(idx_states,1)-1,NFFT/2);
    p_S2  = nan(size(idx_states,1)-1,NFFT/2);
    p_Dia = nan(size(idx_states,1)-1,NFFT/2);
    for row=1:size(idx_states,1)-1
        s1 = PCG(idx_states(row,1):idx_states(row,2));
        s1 = s1.*hamming(length(s1));
        Ft = fft(s1,NFFT);
        p_S1(row,:) = abs(Ft(1:NFFT/2));
        
        sys = PCG(idx_states(row,2):idx_states(row,3));
        sys = sys.*hamming(length(sys));
        Ft  = fft(sys,NFFT);
        p_Sys(row,:) = abs(Ft(1:NFFT/2));
        
        s2 = PCG(idx_states(row,3):idx_states(row,4));
        s2 = s2.*hamming(length(s2));
        Ft = fft(s2,NFFT);
        p_S2(row,:) = abs(Ft(1:NFFT/2));
        
        dia = PCG(idx_states(row,4):idx_states(row+1,1));
        dia = dia.*hamming(length(dia));
        Ft  = fft(dia,NFFT);
        p_Dia(row,:) = abs(Ft(1:NFFT/2));
    end
    P_S1 = nan(1,size(freq_range,1));
    P_Sys = nan(1,size(freq_range,1));
    P_S2 = nan(1,size(freq_range,1));
    P_Dia = nan(1,size(freq_range,1));
    for bin=1:size(freq_range,1)
        idx = (f>=freq_range(bin,1)) & (f<freq_range(bin,2));
        P_S1(1,bin) = median(median(p_S1(:,idx)));
        P_Sys(1,bin) = median(median(p_Sys(:,idx)));
        P_S2(1,bin) = median(median(p_S2(:,idx)));
        P_Dia(1,bin) = median(median(p_Dia(:,idx)));
    end
    features = [P_S1, P_Sys, P_S2, P_Dia];
end

'''