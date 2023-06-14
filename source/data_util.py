'''
utility function to read data, integration, and plot
@author: Muhammad Shahid
'''

import os
import sys
import numpy as np
import math
from scipy.fftpack import fft
from scipy import signal
from scipy.signal import butter, filtfilt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.signal import butter, sosfilt
from scipy import signal
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Replace 'TkAgg' with a different backend
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler



currentpath = os.getcwd()
BasePath = 'D:/DSPLab/Development/SMRTPNT/'
PathDataFolder = '../HospitalData/SmartPantTestData/'  # SmartPantTrainData/'
colmns = 'abcdefghijklmnt'

##### Data formate for signals

DataColumns = ['FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','P7','P8','Fz','Cz','Pz','FC1','FC2','CP1','CP2','FC5','FC6','CP5','CP6','EMG1','EMG2','IO','EMG3','EMG4',
              'LShankACCX','LShankACCY','LShankACCZ','LShankGYROX','LShankGYROY','LShankGYROZ','NC1','RShankACCX','RShankACCY','RShankACCZ','RShankGYROX',
              'RShankGYROY','RShankGYROZ','NC2','WaistACCX','WaistACCY','WaistACCZ','WaistGYROX','WaistGYROY','WaistGYROZ','NC3','ArmACCX','ArmACCY','ArmACCZ',
              'ArmGYROX','ArmGYROY','ArmGYROZ','SC','Label']# 500Hz
EEG_index = np.arange(0,25)
EMG_index = np.arange(25,30)
Tibia_index = np.arange(30,36)
EEGColumns = [DataColumns[index] for index in EEG_index] #DataColumns[EEG_index] # Brain Signals
EMGColumns = [DataColumns[index] for index in EMG_index]#DataColumns[EMG_index] # Muscle signals
TibiaColumns = [DataColumns[index] for index in Tibia_index]#DataColumns[Tibia_index] #Left Shankbone or tibia
ScColumns = DataColumns[57] #skin conductance
LblColumns = DataColumns[58] #Label 0 No Freezing 1 Freezing of Gait


fs = 500#200
cutoff = 6
order = 2
f =19
# Z-score normalization
scaler = StandardScaler()
###########Filtering##############filters
def remove_drift(signal1, fs):
    b, a = signal.butter( 3, 2, 'highpass', fs=fs)
    return signal.filtfilt(b, a, signal1)

def notch(signal1, freq, sample_frequency):
    b, a = signal.iirnotch(freq, 15, sample_frequency)
    return signal.filtfilt(b, a, signal1)

def notch_harmonics(signal, freq, sample_frequency):
    for harmonic in range(1,5): #5
        signal = notch(signal, freq*harmonic, sample_frequency)
    return signal

def apply_to_all(function, signal_array, *args, **kwargs):
    results = []
    for i in range(signal_array.shape[1]):
        results.append(function(signal_array[:,i], *args, **kwargs))
    return np.stack(results, 1)
#####################

def Label_Rolling_Window(label,window_size = 300, stride=30):
    # Generate sample time series data
    time_series = pd.Series(label)

    # Define window size and stride length
    #window_size = 3
    #stride = 2

    # Apply rolling window with strides
    rolling_windows = time_series.rolling(window=window_size)

    # Iterate over the rolling windows
    Data_indxd = []
    for i in range(0, len(time_series)-window_size, stride):
        Window = lambda x: x.tolist()[i:i+window_size] if len(x.tolist()) >= i+window_size else None
        window_data = Window(time_series)#rolling_windows.apply()
        #print(f"Window {i//stride+1}:")
        #print(window_data)
        strd_idx = i
        end_idx = i+window_size
        lables = 0 if np.array(window_data).sum()<240 else 1
        Data_indxd.append((lables, strd_idx,end_idx))
    print(f"Window process completed:")
    return Data_indxd



def find_duplicates(lst):
    duplicates = set()
    unique_elements = set()
    for element in lst:
        if element in unique_elements:
            duplicates.add(element)
        else:
            unique_elements.add(element)
    return list(duplicates)
#####Load Data
def LoadLabelSeries(TasksList):
    tempFrame1 = pd.read_csv(TasksList, names=list(DataColumns)) # Data format=> Acc(x,y,x),Gyro(x,y,z),Meg(x,y,z),feet press(Toe,extr, enter, heel)
    label = tempFrame1['Label'].values
    return label
def loadDataItem(tple):
    # load a csv file, select portion of data, select modality
    tempFrame1 = pd.read_csv(tple[1], names=list(DataColumns))
    data = tempFrame1[EMGColumns].values[tple[0][1]:tple[0][2]]
    #x = apply_to_all(notch_harmonics, x, 60, 1000)
    if(len(data)<1500):
        print('here is the error',tple)
    x = apply_to_all(notch_harmonics, data, 50, 500)
    x = apply_to_all(remove_drift, x, 500)
    lbl  =  tple[0][0]
    return (lbl,x)



#######################Select data from pdframes
def SelectItem(tple,FrameList):
    # load a csv file, select portion of data, select modality
    Anno,file_indx = tple
    data_EEG = FrameList[file_indx][Anno[1]:Anno[2],EEG_index]
    data_Emg = FrameList[file_indx][Anno[1]:Anno[2],EMG_index]
    data_Imu = FrameList[file_indx][Anno[1]:Anno[2],Tibia_index]
    if(len(data_Emg)<300):
        print('here is the error',tple)
    x_Imu = data_Imu
    x_lbl  =  np.array(Anno[0])
    lbl_Emg_Imu = {'lbl':x_lbl,'EEG':data_EEG, 'Emg':data_Emg,'Imu':x_Imu}
    return lbl_Emg_Imu#(lbl,x)


#######################


def rolling_window(a, window_size):
    shape = (a.shape[0] - window_size, window_size) + a.shape[1:]
    strides = (a.strides[0],) + a.strides
    print(shape,strides)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

################################

def median_filter(data, f_size):
    lgth, num_signal = data.shape
    f_data = np.zeros([lgth, num_signal])
    for i in range(num_signal):
        f_data[:, i] = signal.medfilt(data[:, i], f_size)
    return f_data

def meanfltr(data, window):
    lgth, num_signal = data.shape
    f_data = np.zeros([lgth, num_signal])
    for i in range(num_signal):
        f_data[:, i] = np.convolve(data[:,i], np.ones(window)/window, mode='same')#signal.medfilt(data[:, i], f_size)
    return f_data
def mean_filter(data, window_size):

    # Apply moving average filter using np.mean
    # Apply moving max filter using np.maximum.reduce
    filtered_data = np.mean.reduce([data[i:i+window_size] for i in range(data.shape[0]-window_size+1)])
    f_data = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    return f_data


def freq_filter(data, f_size, cutoff):
    lgth, num_signal = data.shape
    f_data = np.zeros([lgth, num_signal])
    lpf = signal.firwin(f_size, cutoff, window='hamming')
    for i in range(num_signal):
        f_data[:, i] = signal.convolve(data[:, i], lpf, mode='same')
    return f_data


def butter_lowpass_filter(data, cutoff, fs, order):
    lgth, num_signal = data.shape
    f_data = np.zeros([lgth, num_signal])
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    for i in range(num_signal):
        f_data[:, i] = filtfilt(b, a, data[:, i])
    return f_data
def bandpass_filter(Data, lowcut, highcut, inplace=False, fs=200, order=7):
    nyq = fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = signal.butter(order, (low, high), btype='band', analog=False, output='sos')
    filtered_signal = np.array([sosfilt(sos, Data[:, i]) for i in range(Data.shape[1])])
    return filtered_signal.T
def bandPass_multiVariat(data,lowcut = .1, highcut=200, fs = 100):
    X_fltr = bandpass_filter(data, lowcut, highcut, inplace=False, fs=fs, order=7)
    avg_x = np.average(X_fltr, axis=1)
    X = X_fltr #- avg_x[:, None] # re-referencing the signal

    return X



def plotData(data, device_indx,
             pressur=0):  # 1= left shin, 2= left thigh, 3 = right shin, 4= right thigh, pressur = 1/0
    plt.figure(figsize=(15, 3))
    plt.subplot(3, 1, 1)
    if device_indx:
        plt.title("Left Shinbone Accelro")
    else:
        plt.title("Right Shinbone Accelro")
    #################
    plt.plot(range(0, len(data[:, 0])), data[:, 0], color='r', label='Ax')
    plt.plot(range(0, len(data[:, 1])), data[:, 1], color='g', label='Ay')
    plt.plot(range(0, len(data[:, 2])), data[:, 2], color='b', label='Az')
    plt.legend()
    ###############
    plt.subplot(3, 1, 2)
    plt.title("Right Shinbone Gyro")
    plt.plot(range(0, len(data[:, 3])), data[:, 3], color='r', label='Gx')
    plt.plot(range(0, len(data[:, 4])), data[:, 4], color='g', label='Gy')
    plt.plot(range(0, len(data[:, 5])), data[:, 5], color='b', label='Gz')
    plt.legend()
    ###############
    plt.subplot(3, 1, 3)
    plt.title("Right Shinbone Magneto")
    plt.plot(range(0, len(data[:, 6])), data[:, 6], color='r', label='Mx')
    plt.plot(range(0, len(data[:, 7])), data[:, 7], color='g', label='My')
    plt.plot(range(0, len(data[:, 8])), data[:, 8], color='b', label='Mz')
    plt.legend()
    plt.show()
def plotGroundPredctd(g_data,p_data,device_indx):# 0= acclro, 1= gyro, 2 = magneto
    plt.figure(figsize=(15, 3))
    plt.subplot(3, 1, 1)
    indx = 3*device_indx
    if device_indx==0:
        plt.title(" Accelerometer XYZ value plot")
    elif device_indx==1 :
        plt.title("Gyroscope XYZ value")
    else :
        plt.title("Magneto XYZ value")
    #################
    plt.plot(range(0, len(g_data[:, indx])), g_data[:, indx], color='g', label='Ag_x')
    plt.plot(range(0, len(p_data[:, indx])), p_data[:, indx], color='r', label='Ap_x')
    plt.legend()
    ###############
    plt.subplot(3, 1, 2)
    plt.plot(range(0, len(g_data[:, indx+1])), g_data[:, indx+1], color='g', label='Ag_y')
    plt.plot(range(0, len(p_data[:, indx+1])), p_data[:, indx+1], color='r', label='Ap_y')
    plt.legend()
    ###############
    plt.subplot(3, 1, 3)
    plt.plot(range(0, len(g_data[:, indx+2])), g_data[:, indx+2], color='g', label='Ag_z')
    plt.plot(range(0, len(p_data[:, indx+2])), p_data[:, indx+2], color='r', label='Ap_z')
    plt.legend()
    plt.legend()
    plt.show()
def plotPressure(g_data,p_data,device_indx):# 0= acclro, 1= gyro, 2 = magneto
    dataLen =  g_data.shape[0]
    max_time = dataLen/fs
    time_steps = np.linspace(0, max_time, dataLen)
    plt.figure(figsize=(15, 4))
    plt.subplot(4, 1, 1)
    indx = 4*device_indx
    if device_indx==0:
        plt.title(" Pressure on Left Foot")
    else :
        plt.title(" Pressure on Right Foot")
    #################
    plt.plot(time_steps, g_data[:, indx], color='g', label='Pg_Toe')
    plt.plot(time_steps, p_data[:, indx], color='r', label='Pp_Toe')
    plt.xlabel("Time [s]")
    plt.ylabel("Magnitude")
    plt.legend()
    ###############
    plt.subplot(4, 1, 2)
    plt.plot(time_steps, g_data[:, indx+1], color='g', label='Pg_Extr')
    plt.plot(time_steps, p_data[:, indx+1], color='r', label='Pp_Extr')
    plt.xlabel("Time [s]")
    plt.ylabel("Magnitude")
    plt.legend()
    ###############
    plt.subplot(4, 1, 3)
    plt.plot(time_steps, g_data[:, indx+2], color='g', label='Pg_Intr')
    plt.plot(time_steps, p_data[:, indx+2], color='r', label='Pp_Intr')
    plt.xlabel("Time [s]")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.subplot(4, 1, 4)
    plt.plot(time_steps, g_data[:, indx+3], color='g', label='Pg_Heel')
    plt.plot(time_steps, p_data[:, indx+3], color='r', label='Pp_Heel')
    plt.xlabel("Time [s]")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.show()
############################
def plotPressureDstribution(g_data,p_data,device_indx):# 0= acclro, 1= gyro, 2 = magneto
    dataLen =  g_data.shape[0]
    max_time = dataLen/fs
    time_steps = np.linspace(0, max_time, dataLen)
    plt.figure(figsize=(15, 4))
    plt.subplot(4, 1, 1)
    indx = 4*device_indx
    if device_indx==0:
        plt.title(" Pressure on Left Foot")
    else :
        plt.title(" Pressure on Right Foot")
    #################
    plt.plot(time_steps, g_data[:, indx], color='g', label='Pg_Toe')
    plt.plot(time_steps, p_data[:, indx], color='r', label='Pp_Toe')
    plt.xlabel("Time [s]")
    plt.ylabel("Magnitude")
    plt.ylim([-0.1,1.1])
    plt.legend()
    ###############
    plt.subplot(4, 1, 2)
    plt.plot(time_steps, g_data[:, indx+1], color='g', label='Pg_Extr')
    plt.plot(time_steps, p_data[:, indx+1], color='r', label='Pp_Extr')
    plt.xlabel("Time [s]")
    plt.ylabel("Magnitude")
    plt.ylim([-0.1,1.1])
    plt.legend()
    ###############
    plt.subplot(4, 1, 3)
    plt.plot(time_steps, g_data[:, indx+2], color='g', label='Pg_Intr')
    plt.plot(time_steps, p_data[:, indx+2], color='r', label='Pp_Intr')
    plt.xlabel("Time [s]")
    plt.ylabel("Magnitude")
    plt.ylim([-0.1,1.1])
    plt.legend()
    plt.subplot(4, 1, 4)
    plt.plot(time_steps, g_data[:, indx+3], color='g', label='Pg_Heel')
    plt.plot(time_steps, p_data[:, indx+3], color='r', label='Pp_Heel')
    plt.xlabel("Time [s]")
    plt.ylabel("Magnitude")
    plt.ylim([-0.1,1.1])
    plt.legend()
    plt.show()

###########################
def plotPressureGrnd(g_data,device_indx):# 0= acclro, 1= gyro, 2 = magneto
    dataLen =  g_data.shape[0]
    max_time = dataLen/fs
    time_steps = np.linspace(0, max_time, dataLen)
    plt.figure(figsize=(15, 4))
    plt.subplot(4, 1, 1)
    indx = 4*device_indx
    if device_indx==0:
        plt.title(" Pressure on Left Foot")
    else :
        plt.title(" Pressure on Right Foot")
    #################
    plt.plot(time_steps, g_data[:, indx], color='g', label='Pg_Toe')
    plt.xlabel("Time [s]")
    plt.ylabel("Magnitude")
    plt.legend()
    ###############
    plt.subplot(4, 1, 2)
    plt.plot(time_steps, g_data[:, indx+1], color='g', label='Pg_Extr')
    plt.xlabel("Time [s]")
    plt.ylabel("Magnitude")
    plt.legend()
    ###############
    plt.subplot(4, 1, 3)
    plt.plot(time_steps, g_data[:, indx+2], color='g', label='Pg_Intr')
    plt.xlabel("Time [s]")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.subplot(4, 1, 4)
    plt.plot(time_steps, g_data[:, indx+3], color='g', label='Pg_Heel')
    plt.xlabel("Time [s]")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.show()
###########################
def traintestsplit(AllpatientsData, train_idx, test_idx):
    trainData = [AllpatientsData[i] for i in train_idx]
    testData = [AllpatientsData[i] for i in test_idx]
    # unrap the lists
    TrainData = [item for sublist in trainData for item in sublist]
    TestData = [item for sublist in testData for item in sublist]
    return TrainData, TestData


################################


def traintestdevision(AllpatientsData, train_idx, test_idx):
    trainData = [AllpatientsData[i] for i in train_idx]
    testData = [AllpatientsData[i] for i in test_idx]
    # unrap the lists
    TrainData = [item for sublist in trainData for item in sublist]
    TestData = [item for sublist in testData for item in sublist]
    return TrainData, TestData


def ExercisListtoPDFrame(SingleActivityseries):
    Left_Shin = SingleActivityseries[0]
    Left_Thigh = SingleActivityseries[1]
    Right_Shin = SingleActivityseries[2]
    Right_Thigh = SingleActivityseries[3]
    pdFram = np.concatenate((Left_Shin[:, 0:9], Left_Thigh[:, 0:9], Right_Shin[:, 0:9], Right_Thigh[:, 0:9],
                             Left_Shin[:, 9:13], Right_Shin[:, 9:13]), axis=1)  # 24 values for acc gyro 8 for pressure
    return pdFram  # left shin=13 left thigh=9,,,Right shin=13 Right thigh=9


def dataNormalization(Finalpdfram):
    # Normalize accelerometer data
    # x = (x-x.mean())/(x.std())
    for i in range(0, Finalpdfram.shape[1]):
        x = Finalpdfram[:, i]
        x_mean = np.average(x)
        x_std = np.std(x)
        #x = (x - x.min()) / (x.max() + 1 - x.min())
        x = (x - x_mean) / (x_std+0.00025)#(x.std()+0.00025)#
        Finalpdfram[:, i] = np.round(x, 5)
    return Finalpdfram
def dataNorml0to1(Finalpdfram):
    # Normalize accelerometer data
    # x = (x-x.mean())/(x.std())
    for i in range(0, Finalpdfram.shape[1]):
        x = Finalpdfram[:, i]
        #x_mean = np.average(x)
        #x_std = np.std(x)
        x = (x - x.min()) / (x.max() + 1 - x.min())
        #x = (x - x_mean) / (x_std+0.00025)#(x.std()+0.00025)#
        Finalpdfram[:, i] = np.round(x, 5)
    return Finalpdfram


def TimeSynch(dataFramList):
    LeftShin = []
    LeftThigh = []
    RightShin = []
    RightThigh = []
    Timeseries = []
    for x in range(0, len(dataFramList), 4):
        dataFrame1 = dataFramList[x]
        dataFrame2 = dataFramList[x + 1]
        dataFrame3 = dataFramList[x + 2]
        dataFrame4 = dataFramList[x + 3]
        ## time synchronization Part comes here

        timeStamp1 = [float(item) for item in dataFrame1[list('t')].values]
        timeStamp2 = [float(item) for item in dataFrame2[list('t')].values]
        timeStamp3 = [float(item) for item in dataFrame3[list('t')].values]
        timeStamp4 = [float(item) for item in dataFrame4[list('t')].values]
        timestampT = np.union1d(timeStamp1, timeStamp3)  # )[::2] downsampling
        timestamp = [item for item in range(int(min(timestampT)), int(max(timestampT)), 10)] #5
        # compute sampling frequency
        fs = int(np.floor(np.mean(1. / (0.005)))) # sampling frequency 200
        #fs = int(np.floor(np.mean(1. / np.diff(timestamp))))

        # timeResample = timstamp[::2]
        for c in selected_colmns_shin:  # for loop for shinbone
            float_valueF1 = [int(item) for item in dataFrame1[list(c)].values]
            float_valueF3 = [int(item) for item in dataFrame3[list(c)].values]
            LeftShin.append(np.interp(np.array(timestamp), np.array(timeStamp1), np.array(float_valueF1)))
            RightShin.append(np.interp(np.array(timestamp), np.array(timeStamp3), np.array(float_valueF3)))
        for c in selected_colmns_thigh:  # for loop for thighbone
            float_valueF2 = [int(item) for item in dataFrame2[list(c)].values]
            float_valueF4 = [int(item) for item in dataFrame4[list(c)].values]
            # interpolation for value calculation
            LeftThigh.append(np.interp(np.array(timestamp), np.array(timeStamp2), np.array(float_valueF2)))
            RightThigh.append(np.interp(np.array(timestamp), np.array(timeStamp4), np.array(float_valueF4)))
    Timeseries.append(np.transpose(np.row_stack(LeftShin)))
    Timeseries.append(np.transpose(np.row_stack(LeftThigh)))
    Timeseries.append(np.transpose(np.row_stack(RightShin)))
    Timeseries.append(np.transpose(np.row_stack(RightThigh)))
    return Timeseries

def LoadDataFile(Tasktxt):
    tempFrame1 = pd.read_csv(Tasktxt, names=list(DataColumns)) # Data format=> Acc(x,y,x),Gyro(x,y,z),Meg(x,y,z),feet press(Toe,extr, enter, heel)
    label = tempFrame1['Label'].values
    # # for Files in filesList : #range(0,len(fileList),1):
    # for i, file in enumerate(TasksList):
    #     # print('file name',file)
    #     tempFrame1 = pd.read_csv(file, names=list(colmns)).fillna(
    #         0)  # Data format=> Acc(x,y,x),Gyro(x,y,z),Meg(x,y,z),feet press(Toe,extr, enter, heel)
    #     tempFrame = (tempFrame1.loc[~(tempFrame1 == 0).all(axis=1)]).copy()  # remove the rows with NA values
    #     t = round(tempFrame['t'] / 10000)
    #     tempFrame['t'] = t.astype(int)
    #     if i == 0 or i == 2:
    #         temp2 = tempFrame[selected_colmns_shin]
    #     else:
    #         temp2 = tempFrame[selected_colmns_thigh]
    #     dataFrames.append(temp2)  # .to_numpy()  # Left shinnbone
    #
    # Timeseries = TimeSynch(dataFrames)
    # Finalpdframe = ExercisListtoPDFrame(Timeseries)
    # os.chdir(currentpath)
    return label


def LoadLabelSeries(TasksList):
    dataFrame = pd.read_csv(TasksList, names=list(DataColumns)) # Data format=> Acc(x,y,x),Gyro(x,y,z),Meg(x,y,z),feet press(Toe,extr, enter, heel)
    label = dataFrame['Label'].values
    Data = dataFrame.drop(LblColumns, axis=1)
    Data = scaler.fit_transform(Data.to_numpy())
    Data_filtr = butter_lowpass_filter(Data, 100, fs, 5)
    Data_filtr = Data_filtr[::5,:] #
    #####Band pass filtering EEG+EMG, IMU signals, inertial 0.5–16 Hz EMG 1.6 to 30 Hz ### normalization of signal
    Data_filtr[EEG_index]= bandPass_multiVariat(Data_filtr[EEG_index],lowcut = 1.6, highcut=30, fs = 100)
    Data_filtr[EMG_index]= bandPass_multiVariat(Data_filtr[EMG_index],lowcut = 1.6, highcut=30, fs = 100)
    Data_filtr[Tibia_index]= bandPass_multiVariat(Data_filtr[Tibia_index],lowcut = 0.5, highcut=16, fs = 100)

    label = label[::5]
    return (label,Data_filtr)


def ApplyNormlzationFiltering(Dataframe):
    TempFrame = Dataframe.copy()

    # Filter the data, and plot both the original and filtered signals.
    fltrd = butter_lowpass_filter(TempFrame, cutoff, fs, order)
    y = median_filter(fltrd, f)
    # apply mu law quantization to reduce level
    #y[:,36:44] = apply_muLaw(y[:,36:44],256)
    DataFram = dataNormalization(y)

    return DataFram
def ApplyNormlzationAndFiltering(Dataframe):
    TempFrame = Dataframe.copy()
    DataFram = dataNormalization(TempFrame)
    # Filter the data, and plot both the original and filtered signals.
    DataFram = butter_lowpass_filter(DataFram, cutoff, fs, order)
    y = median_filter(DataFram, 21)
    return y
def ApplyNormFltng(AllDatafrmList):
    FltrdDatafrmList = []
    for Dataframe in AllDatafrmList:
        FltrdDatafrmList.append(ApplyNormlzationFiltering(Dataframe))

    return FltrdDatafrmList
def DFrametoXY(DframList):
    ndDataArray = np.concatenate(DframList,axis=0)
    Y = ndDataArray[:,36:44]
    X = ndDataArray[:,0:36]
    return X,Y
def getBatch(TrainfltrData,batch_idx):
    Dfram= TrainfltrData[batch_idx]
    Dfram_y = Dfram[:,36:44]
    Dfram_x = Dfram[:,0:36]
    return Dfram_x,Dfram_y
def getBatches(TrainfltrData,batch_idx):
    trainDatalist = [TrainfltrData[i] for i in batch_idx]
    Dfram = np.concatenate(trainDatalist,axis=0)
    Dfram_y = Dfram[:,36:44]
    Dfram_x = Dfram[:,0:36]
    return Dfram_x,Dfram_y
def getBatchesP(TrainfltrData,batch_idx): # get batch with label as pressure
    trainDatalist = [TrainfltrData[i] for i in batch_idx]
    Dfram = np.concatenate(trainDatalist,axis=0)
    Dfram_x = Dfram[:,0:36]
    Dfram_y = Dfram[:,36:44]
    return Dfram_x,Dfram_y
# 2D time series data to 3D tensor transformation of data
def smartpant3DTensor(X):
    x_t = X.reshape(-1, 4, 9)  # np.empty((0, 24), float)
    #y_t = y.reshape(-1, 2, 3)  # np.empty((0, 8), float)

    return x_t
