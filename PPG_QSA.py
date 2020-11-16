# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 11:23:16 2020

@author: aymahm
"""
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.signal import butter, filtfilt 
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from scipy.stats import entropy
import time
import os
from entropy import spectral_entropy


path = 'C:/Users/aymahm/Desktop/Project/'
Frequency_rate = 20
##################Segmentation#################
def readData_Seg(): 
    Frequency_rate = 20
    path = 'C:/Users/aymahm/Desktop/Project/'
    for j in range(11,20):
        all_files_dataset1 = glob.glob(path+'Watch_data/ID'+str(j)+'/Watch/SixDay/data_*.csv')
        for filename in all_files_dataset1:
            dataset1 = pd.read_csv(filename, sep='\t', usecols=['timestamp', 'ppg', 'hrm','accx','accy','accz','grax','gray','graz','gyrx','gyry','grz', 'pressure', 'stress'])
            for i in range(0,int(len(dataset1)/(Frequency_rate*30))):
                df = pd.DataFrame(dataset1[i*30*Frequency_rate:(i+1)*30*Frequency_rate])
                localtime = dataset1.loc[(i*30*Frequency_rate)+1,'timestamp']
                filename_segment = path+'Segmentation/ID'+str(j)+'/SixDay_30s/'+"T-"+str(localtime)+".csv"
                df.to_csv(filename_segment, index=False)
                  
#readData_Seg()        
##############################################################################

###############Bandpass Filter###################
def BandpassFiilter(data, fs): 
    def butter_bandpass(lowcut, highcut, fs, order=3):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    lowcut = 40/60  ####40
    highcut = 180/60  ###180

    #print(data)
    ppg_sig = data['ppg']
    ACC_x = data['accx']
    ACC_y = data['accy']
    ACC_z = data['accz']
    GYR_x = data['gyrx']
    GYR_y = data['gyry']
    GYR_z = data['grz']
# Filter the noisy signal.
    ppg_filt = butter_bandpass_filter(ppg_sig, lowcut, highcut, fs, order=3)
    ACC_x_filt = butter_bandpass_filter(ACC_x, lowcut, highcut, fs, order=3)
    ACC_y_filt = butter_bandpass_filter(ACC_y, lowcut, highcut, fs, order=3)
    ACC_z_filt = butter_bandpass_filter(ACC_z, lowcut, highcut, fs, order=3)
    GYR_x_filt = butter_bandpass_filter(GYR_x, lowcut, highcut, fs, order=3)
    GYR_y_filt = butter_bandpass_filter(GYR_y, lowcut, highcut, fs, order=3)
    GYR_z_filt = butter_bandpass_filter(GYR_z, lowcut, highcut, fs, order=3)
    df_data = list(zip(*[ppg_filt, ACC_x_filt, ACC_y_filt, ACC_z_filt, GYR_x_filt, GYR_y_filt, GYR_z_filt]))
    df = pd.DataFrame(df_data, columns=['ppg_filt', 'ACC_x_filt', 'ACC_y_filt', 'ACC_z_filt', 'GYR_x_filt', 'GYR_y_filt', 'GYR_z_filt'])
    return df

##############Filter PPG only#############
def BandpassFiilter_PPG(data, fs): 
    def butter_bandpass(lowcut, highcut, fs, order=3):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    lowcut = 40/60  ####40
    highcut = 180/60  ###180

    ppg_sig = data['ppg']
    
# Filter the noisy signal.
    ppg_filt = butter_bandpass_filter(ppg_sig, lowcut, highcut, fs, order=3)
    df_data = list(zip(*[ppg_filt]))
    df = pd.DataFrame(df_data, columns=['ppg_filt'])
    return df

#############Peak detection########################## 
def _Peak_detection(sigdata, signalfilt, fs):
    NN_index_sig = np.array(signal.argrelextrema(signalfilt, np.greater)).reshape(1,-1)[0]
    f, ppg_den = signal.periodogram(signalfilt, fs)
    min_f = np.where(f >= 0.6)[0][0] 
    max_f = np.where(f >= 3.0)[0][0] 
    ppgHRfreq = ppg_den[min_f:max_f]
    HRfreq = f[min_f:max_f]    
    HRf = HRfreq[np.argmax(ppgHRfreq)]
    boundary = 0.5
    if HRf - boundary > 0.6:
        HRfmin = HRf - boundary
    else:
        HRfmin = 0.6
    if HRf + boundary < 3.0:
        HRfmax = HRf + boundary
    else:
        HRfmax = 3.0
    filtered = _ButterFilt(signalfilt,fs,np.array([HRfmin,HRfmax]),5,'bandpass')
    NN_index_filtered = np.array(signal.argrelextrema(filtered, np.greater)).reshape(1,-1)[0]
    rpeak = np.array([]).astype(int)
    for i in NN_index_filtered:
        rpeak = np.append(rpeak,NN_index_sig[np.abs(i - NN_index_sig).argmin()])
    rpeak = np.unique(rpeak)
    return(rpeak)

def _segmentation_heartCycle(sigfile,filtsig,NN_index):
    MM_index = np.array([]).astype(int)
    for i in range(NN_index.shape[0]-1):
        MM_index = np.append(MM_index,np.argmin(filtsig[NN_index[i]:NN_index[i+1]]) + NN_index[i])
    return(MM_index)
    
def _ButterFilt(sig,fs,fc,order,btype):
    w = fc/(fs/2)
    b, a = signal.butter(order, w, btype =btype, analog=False)
    filtered = signal.filtfilt(b, a, sig)
    return(filtered)
    
def _range(x):
    r = np.round((np.amax(x)-np.amin(x)),3)
    return(r)
#################Shannon Entropy###############
def _Shannon_Entropy(signalfilt):
    Hist = np.histogram(signalfilt)[0]
    prob = np.array([])
    for i in range(len(Hist)):
        prob = Hist/Hist.sum()
    prob_array = np.array(prob)
    return entropy(prob_array,base=2)
###############Aproximate Entropy###################################
def _aprox_Entropy(U, m, r) -> float:
    U = np.array(U)
    N = U.shape[0]
            
    def _phi(m):
        z = N - m + 1.0
        x = np.array([U[i:i+m] for i in range(int(z))])
        X = np.repeat(x[:, np.newaxis], 1, axis=2)
        C = np.sum(np.absolute(x - X).max(axis=2) <= r, axis=0) / z
        return np.log(C).sum() / z
    
    return abs(_phi(m + 1) - _phi(m))

########################Feature Extraction############################################
def main():
    all_files_dataset1 = glob.glob(path+'Segmentation/ID12/SixDay_30s/'+'T-*.csv')
    sigdata = []
    features1 = []
    features2 = []
    features3 = []
    features4 = []
    features5 = []
    features6 = []
    features7 = []
    features8 = []
    features9 = []
    features10 = []
    features11 = []
    features12 = []
    features13 = []
    features14 = []
    features15 = []
    features16 = []
    features17 = []
    features18 = []
    features19 = []
    features20 = []
    features21 = []
    features22 = []
    features23 = []
    features24 = []
    timestamp = []

    for filename in all_files_dataset1:
        sigdata = pd.read_csv(filename, sep=',')
        head, tail = os.path.split(filename)
        timestamp.append(tail)
        signalfilt = np.array(BandpassFiilter(sigdata, Frequency_rate)['ppg_filt'])
        
        rpeaks = _Peak_detection(sigdata, signalfilt, Frequency_rate)        
        
        df_features = pd.DataFrame(columns=['skewness', 'kurtosis', 'approxentro'])
        heartpeak = _segmentation_heartCycle(sigdata,signalfilt,rpeaks)
        for i in range(heartpeak.shape[0]-1):
            heart_cycle =  signalfilt[heartpeak[i]:heartpeak[i+1]]
            f_skew = stats.skew(heart_cycle)
            f_kurt = stats.kurtosis(heart_cycle)
            f_appentropy = _aprox_Entropy(heart_cycle, 2, 7)
            df_features.loc[len(df_features)] = [f_skew,f_kurt,f_appentropy]
        features1.append(np.mean(signalfilt))
        features2.append(np.std(signalfilt))
        features3.append(np.median(signalfilt))
        features4.append(_range(df_features['skewness']))
        features5.append(_range(df_features['kurtosis']))
        features6.append(_range(df_features['power']))
        features7.append(_range(df_features['approxentro']))
        features8.append(_Shannon_Entropy(signalfilt))
        features9.append(_aprox_Entropy(signalfilt, 2, 7))

        
        frequency_psd = signal.periodogram(signalfilt, fs= Frequency_rate)[0]
        amplitude_psd = signal.periodogram(signalfilt, fs= Frequency_rate)[1]
        
        features10.append(np.trapz(amplitude_psd[np.where(np.logical_and(frequency_psd >= 0.6, frequency_psd <= 0.8))], frequency_psd[np.where(np.logical_and(frequency_psd >= 0.6, frequency_psd <= 0.8))]))  # between 0.6 to 0.8
        features11.append(np.trapz(amplitude_psd[np.where(np.logical_and(frequency_psd >= 0.8, frequency_psd <= 1))], frequency_psd[np.where(np.logical_and(frequency_psd >= 0.8, frequency_psd <= 1))]))  # between 0.8 to 1
        features12.append(np.trapz(amplitude_psd[np.where(np.logical_and(frequency_psd >= 1, frequency_psd <= 1.2))], frequency_psd[np.where(np.logical_and(frequency_psd >= 1, frequency_psd <= 1.2))]))  # between 1 to 1.2
        features13.append(np.trapz(amplitude_psd[np.where(np.logical_and(frequency_psd >= 1.2, frequency_psd <= 1.4))], frequency_psd[np.where(np.logical_and(frequency_psd >= 1.2, frequency_psd <= 1.4))]))  # between 1.2 to 1.4
        features14.append(np.trapz(amplitude_psd[np.where(np.logical_and(frequency_psd >= 1.4, frequency_psd <= 1.6))], frequency_psd[np.where(np.logical_and(frequency_psd >= 1.4, frequency_psd <= 1.6))]))  # between 1.4 to 1.6
        features15.append(np.trapz(amplitude_psd[np.where(np.logical_and(frequency_psd >= 1.6, frequency_psd <= 1.8))], frequency_psd[np.where(np.logical_and(frequency_psd >= 1.6, frequency_psd <= 1.8))]))  # between 1.6 to 1.8
        features16.append(np.trapz(amplitude_psd[np.where(np.logical_and(frequency_psd >= 1.8, frequency_psd <= 2))], frequency_psd[np.where(np.logical_and(frequency_psd >= 1.8, frequency_psd <= 2))]))  # between 1.8 to 2
        features17.append(np.trapz(amplitude_psd[np.where(np.logical_and(frequency_psd >= 2, frequency_psd <= 2.2))], frequency_psd[np.where(np.logical_and(frequency_psd >= 2, frequency_psd <= 2.2))]))  # between 2 to 2.2
        features18.append(np.trapz(amplitude_psd[np.where(np.logical_and(frequency_psd >= 2.2, frequency_psd <= 2.4))], frequency_psd[np.where(np.logical_and(frequency_psd >= 2.2, frequency_psd <= 2.4))]))  # between 2.2 to 2.4
        features19.append(np.trapz(amplitude_psd[np.where(np.logical_and(frequency_psd >= 2.4, frequency_psd <= 2.6))], frequency_psd[np.where(np.logical_and(frequency_psd >= 2.4, frequency_psd <= 2.6))]))  # between 2.4 to 2.6
        features20.append(np.trapz(amplitude_psd[np.where(np.logical_and(frequency_psd >= 2.6, frequency_psd <= 2.8))], frequency_psd[np.where(np.logical_and(frequency_psd >= 2.6, frequency_psd <= 2.8))]))  # between 2.6 to 2.8
        features21.append(np.trapz(amplitude_psd[np.where(np.logical_and(frequency_psd >= 2.8, frequency_psd <= 3))], frequency_psd[np.where(np.logical_and(frequency_psd >= 2.8, frequency_psd <= 3))]))  # between 2.8 to 3
        features22.append(np.std(amplitude_psd))
        features23.append(np.max(amplitude_psd))
        features24.append(spectral_entropy(signalfilt, Frequency_rate, method='welch'))
        
    df_data = list(zip(*[timestamp, features1, features2, features3, features4, features5, features6, features7, features8, features9, features10, features11, features12, features13, features14, features15, features16, features17, features18, features19, features20, features21, features22, features23, features24]))
    df = pd.DataFrame(df_data, columns=['Timestamp', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18','f19','f20','f21','f22','f23','f24'])
    df = df.dropna()
    #df.to_csv(path+'Features_ID1_30s_40.csv', mode='a', index=False)
######################Normalized######################### 
    #print(df)
    df_z = pd.DataFrame(columns=['ave','std'])    
    df_z['ave'] = ([np.mean(df['f1']),np.mean(df['f2']),np.mean(df['f3']),np.mean(df['f4']),np.mean(df['f5'])])
    df_z['std'] = ([np.std(df['f1']),np.std(df['f2']),np.std(df['f3']),np.std(df['f4']),np.std(df['f5'])])

    df['f1'] = (df['f1'] - df_z['ave'][0])/df_z['std'][0]
    df['f2'] = (df['f2'] - df_z['ave'][1])/df_z['std'][1]
    df['f3'] = (df['f3'] - df_z['ave'][2])/df_z['std'][2]
    df['f4'] = (df['f4'] - df_z['ave'][3])/df_z['std'][3]
    df['f5'] = (df['f5'] - df_z['ave'][4])/df_z['std'][4]
    df['f6'] = (df['f6'] - df_z['ave'][5])/df_z['std'][5]
    df['f7'] = (df['f7'] - df_z['ave'][6])/df_z['std'][6]
    df['f8'] = (df['f8'] - df_z['ave'][7])/df_z['std'][7]
    df['f9'] = (df['f9'] - df_z['ave'][8])/df_z['std'][8]
    df['f10'] = (df['f10'] - df_z['ave'][9])/df_z['std'][9]
    df['f11'] = (df['f11'] - df_z['ave'][10])/df_z['std'][10]
    df['f12'] = (df['f12'] - df_z['ave'][11])/df_z['std'][11]
    df['f13'] = (df['f13'] - df_z['ave'][12])/df_z['std'][12]
    df['f14'] = (df['f14'] - df_z['ave'][13])/df_z['std'][13]
    df['f15'] = (df['f15'] - df_z['ave'][14])/df_z['std'][14]
    df['f16'] = (df['f16'] - df_z['ave'][15])/df_z['std'][15]
    df['f17'] = (df['f17'] - df_z['ave'][16])/df_z['std'][16]
    df['f18'] = (df['f18'] - df_z['ave'][17])/df_z['std'][17]
    df['f19'] = (df['f19'] - df_z['ave'][18])/df_z['std'][18]
    df['f20'] = (df['f20'] - df_z['ave'][19])/df_z['std'][19]
    df['f21'] = (df['f21'] - df_z['ave'][20])/df_z['std'][20]
    df['f22'] = (df['f22'] - df_z['ave'][21])/df_z['std'][21]
    df['f23'] = (df['f23'] - df_z['ave'][22])/df_z['std'][22]
    df['f24'] = (df['f24'] - df_z['ave'][23])/df_z['std'][23]

    df_data = list(zip(*[df['Timestamp'], df['f1'], df['f2'], df['f3'], df['f4'], df['f5'], df['f6'], df['f7'], df['f8'], df['f9'], df['f10'], df['f11'], df['f12'], df['f13'], df['f14'], df['f15'], df['f16'], df['f17'], df['f18'], df['f19'], df['f20'], df['f21'], df['f22'], df['f23'], df['f24']]))
    df_data = pd.DataFrame(df, columns=['Timestamp', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18','f19','f20','f21','f22','f23','f24'])
    df_data.to_csv(path+'norm_features.csv', mode='a', index=False)

    
#main()

##########Plot features##############
def Figure_Diff_Features():
    all_files_dataset1 = glob.glob(path+'Segmentation/ID1/Figure_paper_B/'+'T-*.csv')
    sigdata = []
    features1_1 = np.array([])
    features2_1 = np.array([])
    features3_1 = np.array([])
    features4_1 = np.array([])
    features5_1 = np.array([])
    timestamp = np.array([])
    segment = np.array([])
    for filename in all_files_dataset1:
        sigdata = pd.read_csv(filename, sep=',')
        head, tail = os.path.split(filename)
    ######Filtering##########
        signalfilt = np.array(BandpassFiilter(sigdata, Frequency_rate)['ppg_filt'])
    ########Peak Detection#############
        rpeaks = _Peak_detection(sigdata, signalfilt, Frequency_rate)     
    #########Feature extraction############
        df_features = pd.DataFrame(columns=['skewness', 'kurtosis', 'approxentro'])
        heartpeak = _segmentation_heartCycle(sigdata,signalfilt,rpeaks)
        for i in range(heartpeak.shape[0]-1):
            heart_cycle =  signalfilt[heartpeak[i]:heartpeak[i+1]]
            f_skew = stats.skew(heart_cycle)
            f_kurt = stats.kurtosis(heart_cycle)
            f_appentropy = _aprox_Entropy(heart_cycle, 2, 7) 
            df_features.loc[len(df_features)] = [f_skew,f_kurt,f_appentropy]
    
        timestamp = np.append(timestamp,tail)
        features1_1 = np.append(features1_1,_range(df_features['skewness']))
        features2_1 = np.append(features2_1,_range(df_features['kurtosis']))
        features3_1 = np.append(features3_1,_range(df_features['approxentro']))
        features4_1 = np.append(features4_1,_Shannon_Entropy(signalfilt))
        features5_1 = np.append(features5_1,spectral_entropy(signalfilt, Frequency_rate, method='welch'))
    
    all_files_dataset1 = glob.glob(path+'Segmentation/ID1/Figure_paper_G/'+'T-*.csv')
    sigdata = []
    features1_2 = np.array([])
    features2_2 = np.array([])
    features3_2 = np.array([])
    features4_2 = np.array([])
    features5_2 = np.array([])
    timestamp = np.array([])
    segment = np.array([])
    for filename in all_files_dataset1:
        sigdata = pd.read_csv(filename, sep=',')
        head, tail = os.path.split(filename)
        signalfilt = np.array(BandpassFiilter(sigdata, Frequency_rate)['ppg_filt'])
        rpeaks = _Peak_detection(sigdata, signalfilt, Frequency_rate)        
        df_features = pd.DataFrame(columns=['skewness', 'kurtosis', 'approxentro'])
        heartpeak = _segmentation_heartCycle(sigdata,signalfilt,rpeaks)
        for i in range(heartpeak.shape[0]-1):
            heart_cycle =  signalfilt[heartpeak[i]:heartpeak[i+1]]
            f_skew = stats.skew(heart_cycle)
            f_kurt = stats.kurtosis(heart_cycle)
            f_appentropy = _aprox_Entropy(heart_cycle, 2, 7) 
            df_features.loc[len(df_features)] = [f_skew,f_kurt,f_appentropy]
    
        timestamp = np.append(timestamp,tail)
        features1_2 = np.append(features1_2,_range(df_features['skewness']))
        features2_2 = np.append(features2_2,_range(df_features['kurtosis']))
        features3_2 = np.append(features3_2,_range(df_features['approxentro']))
        features4_2 = np.append(features4_2,_Shannon_Entropy(signalfilt))
        features5_2 = np.append(features5_2,spectral_entropy(signalfilt, Frequency_rate, method='welch'))


    
    segment = [1,2,3,4,5]
    segment2 = [6,7,8,9,10]  
    plt.figure(figsize = [6,2.8])
    plt.scatter(segment,features5_2,c='b',label='Reliable',s=100)
    plt.scatter(segment2,features5_1,c='r',label='Unreliable',s=100)
    plt.yticks(np.arange(0, 7, 1),fontsize=16)
    plt.xticks(np.arange(1, 11,1),fontsize=16)
    #plt.tick_params(length=0.5, width=0.5)
    
    plt.xlabel('#Segment', fontsize=24)
    plt.ylabel('Amplitude', fontsize=24)
    plt.legend(fontsize = 16)
    plt.grid(ls = '-.')
    plt.show    
        
#Figure_Diff_Features()

def plot_signals():
    all_files_dataset1 = glob.glob(path+'Segmentation/ID1/Figure_paper_G/'+'T-*.csv')
    sigdata = []
    for filename in all_files_dataset1:
        sigdata = pd.read_csv(filename, sep=',')
        head, tail = os.path.split(filename)
        signalfilt = np.array(BandpassFiilter(sigdata, Frequency_rate)['ppg_filt'])
        plt.plot(signalfilt)
        plt.show()

#plot_signals()
        
#######Plot Peaks#####
def __plot_peak():
    sigfile = pd.read_csv(path+'Segmentation/ID2/T-*.csv', sep=',')
    filt_sig_sample = np.array(BandpassFiilter(sigfile, Frequency_rate)['ppg_filt'])
    
    sigfile_Peak = pd.read_csv(path+'Segmentation/ID2/T-*.csv', sep=',')
    filt_sig_sample = np.array(BandpassFiilter(sigfile_Peak, Frequency_rate)['ppg_filt'])
    plt.plot(filt_sig_sample)
    rpeaks = _Peak_detection(sigfile_Peak, filt_sig_sample, Frequency_rate) 
    heartpeak = _segmentation_heartCycle(sigfile_Peak,filt_sig_sample,rpeaks)
    
    plt.plot(np.arange(filt_sig_sample.shape[0])[rpeaks],filt_sig_sample[rpeaks],'x')
    plt.plot(np.arange(filt_sig_sample.shape[0])[heartpeak],filt_sig_sample[heartpeak],'x')
    plt.xlabel("Time")
    plt.ylabel("PPG")
    plt.show()
    
#__plot_peak()


