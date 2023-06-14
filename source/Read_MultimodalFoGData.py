import os
import torch
from data_util import *
import random
from torch.utils.data import Dataset, DataLoader
#input_scales = np.logspace(np.log10(1), np.log10(10), 10) # frequencies 16 to 160
input_scales = np.logspace(np.log10(1.625), np.log10(16.25), 10)    #frequencies 10 to 100(1.625)
#input_scales = np.logspace(np.log10(1.625), np.log10(32), 10)    #frequencies 5 to 100(1.625)
#input_scales = np.logspace(np.log10(1.625), np.log10(27), 10)    #frequencies 6 to 100(1.625)
#input_scales = np.logspace(np.log10(1.625), np.log10(20), 10)    #frequencies 8 to 100(1.625)
#input_scales = np.logspace(np.log10(1.48), np.log10(16.25), 10)    #frequencies 10 to 110(1.625)
def normalizeSH(Finalpdfram):
 # Normalize accelerometer data
 #x = (x-x.mean())/(x.std())
 for i in range(0,Finalpdfram.shape[1],1):
     x= Finalpdfram[:,i]
     x = (x-x.min())/(x.max()+1-x.min()) #(x.std()+0.025)#
     Finalpdfram[:, i] = np.round(x,5)
 return Finalpdfram
def indexData(patients):
     window = 300 ##best 450 on pt3
     Annotation = []
     DataFrames  = []
     Y_out = []
     file_idx = 0
     for patient in patients:
         for sngleTask in patient:
             print('exercise name: ',sngleTask)
             Labels,dataFrame = LoadLabelSeries(sngleTask)
             Labels = Label_Rolling_Window(Labels)
             #scalo_V = make_scalogram(Labels,input_scales,window,10)
             Label_list = [(item,file_idx) for item in Labels]
             Annotation.append(Label_list)
             DataFrames.append(dataFrame)
             file_idx+=1
         ###unrap the list
     Annotation = [item for sublist in Annotation for item in sublist] # unrap X
         #Y = np.array([item for sublist in Y_out for item in sublist]) # unrap Y

     return Annotation,DataFrames
def classSegregat(ListData): # return Class0 , Class1 tuples
    Class0 = []
    Class1 = []
    for tpl in ListData:
       if(tpl[0][0]==0):
           Class0.append(tpl)
       else:
           Class1.append(tpl)

    return Class0, Class1




class MMFoGDataset(Dataset):
     def __init__(self, base_dir=None, patient_idx=None, tasks=None):
         directories = []
         self.minLength = 1000 #minimum lenght of time series sequence
         if base_dir is not None:
             self.PatientList = os.listdir(base_dir)
         else:
             print('Base directory is not define')
         listPatients = []
         PatientTasks = []
         for idx in patient_idx:
             path = os.path.join(base_dir,self.PatientList[idx])
             Tasks = os.listdir(path)
             PatientTasks.append([path+'/'+Tasks[idx] for idx in tasks])
             listPatients.append(path)
         self.PatientList= listPatients
         self.PatientsTasks= PatientTasks
         Annotaions,DataFrames = indexData(self.PatientsTasks)
         Class0, Class1 = classSegregat(Annotaions)
         random.shuffle(Annotaions)
         self.Annotations = Annotaions
         self.DataFrames = DataFrames
         class_ratio =  len(Class1)/(len(Class1)+len(Class0))
         print('The class ratio is=',class_ratio)

     def __len__(self):
        return len(self.Annotations)

     def __getitem__(self, i):
         Data = SelectItem(self.Annotations[i],self.DataFrames)
         lbl = Data['lbl']
         raw_Eeg  = Data['EEG']
         raw_Emg  = Data['Emg']/1000
         raw_Imu  = Data['Imu']/1000
         #####Filtering and normalization of signal
         #raw_Imu = raw_Imu/500
         #lbl,raw_emg = loadDataItem(self.Annotations[i])
         #raw_emg = raw_emg / 500 #20
         #raw_emg = torch.tensor(50*np.tanh(raw_emg/50.),dtype=torch.float32)
         raw_eeg = torch.tensor(raw_Eeg,dtype=torch.float32)
         raw_emg = torch.tensor(raw_Emg ,dtype=torch.float32)
         raw_imu = torch.tensor(raw_Imu,dtype=torch.float32)
         label = torch.tensor(lbl,dtype=torch.float)
         return (label,raw_imu)


if __name__ == '__main__':
    #FLAGS(sys.argv)
    train_idx = [0]  #best
    test_idx = [0]  #
    tasks_types = [0]
    BaseDirectory = '../Filtered Data/'
    MultimdlEMG = MMFoGDataset(BaseDirectory,train_idx,tasks_types)
    # train_loader = DataLoader(dataset=MultimdlEMG ,
    #                       batch_size=4,
    #                       shuffle=True,
    #                       num_workers=1)
    # dataiter = iter(train_loader)
    for i in range(10):
        print('item number',i)
        data = MultimdlEMG[i]
        # data = next(dataiter)
        # labels,features = data
        # print(features, labels)





