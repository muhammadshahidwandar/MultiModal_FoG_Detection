import os
import sys
import numpy as np
import logging
import subprocess
import random
import soundfile as sf

import torch
from torch import nn
import torch.nn.functional as F

from Read_MultimodalFoGData import MMFoGDataset

#from data_util import
from sklearn.metrics import precision_score, recall_score, f1_score
####Audio Wave Generation
#from WaveGenmodels import WaveGANGenerator
###wavGen parameters
noise_latent_dim = 128#768 #100  # size of input feature dimention to Wavgenrator
model_capacity_size = 32 #64  # model capacity during training can be reduced to 32 for larger window length of 2 seconds and 4 seconds
audio_len = 16000    #16384#32768
dim_in = 256
num_genrtr_ins = dim_in*dim_in
#emg sequence length
seq_len = 100#200 # sequence length of emg features where emg raw seq_len*8

from absl import flags
FLAGS = flags.FLAGS #'model_size', 768, 128
flags.DEFINE_integer('model_size',128 , 'number of hidden dimensions')
flags.DEFINE_integer('num_layers', 2, 'number of layers') #transformer layers default 6
flags.DEFINE_integer('batch_size', 128, 'training batch size')
flags.DEFINE_integer('epochs', 50, 'number of training epochs')
flags.DEFINE_float('learning_rate', 1e-4, 'learning rate')
flags.DEFINE_integer('learning_rate_patience', 5, 'learning rate decay patience')
flags.DEFINE_integer('learning_rate_warmup', 500, 'steps of linear warmup')
flags.DEFINE_string('start_training_from', None, 'start training from this model') #'./models/transduction_model.pt' ./output/model.pt
flags.DEFINE_float('data_size_fraction', 1.0, 'fraction of training data to use')
flags.DEFINE_float('phoneme_loss_weight', 0.5, 'weight of auxiliary phoneme prediction loss')
flags.DEFINE_float('l2', 1e-5, 'weight decay')
flags.DEFINE_float('dropout', .1, 'dropout')
flags.DEFINE_string('output_directory', 'output', 'output directory')

######TCN parameters
num_channels =  [128, 128, 128]
flags.DEFINE_integer('input_size', 8, 'number of output channels at each TCN block')
flags.DEFINE_integer('num_classes', 100 , 'number of output channels at each TCN block')
#flags.DEFINE_integer('dropout', 0, 'number of output channels at each TCN block')
flags.DEFINE_string('relu_type', 'relu' , 'number of output channels at each TCN block')
flags.DEFINE_bool('dwpw', False , 'number of output channels at each TCN block')

#####resbloclk
class ResBlock(nn.Module):
    def __init__(self, num_ins, hidden_size, stride=1, num_classes=2):
        super().__init__()

        self.conv1 = nn.Conv1d(num_ins, hidden_size, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_size)

        if stride != 1 or num_ins != hidden_size:
            self.residual_path = nn.Conv1d(num_ins, hidden_size, 1, stride=stride)
            self.res_norm = nn.BatchNorm1d(hidden_size)
        else:
            self.residual_path = None


    def forward(self, x):
        input_value = x

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.residual_path is not None:
            res = self.res_norm(self.residual_path(input_value))
        else:
            res = input_value

        return F.relu(x + res)

class Model(nn.Module):
    def __init__(self, num_ins,num_classes):
        super().__init__()
        self.frontend_nout = 32
        self.backend_out = 32
        self.conv_blocks = nn.Sequential(
            ResBlock(num_ins, FLAGS.model_size, 2),
            #ResBlock(FLAGS.model_size, FLAGS.model_size, 1),
            ResBlock(FLAGS.model_size, FLAGS.model_size, 2),
        )
        #encoder_layer = TransformerEncoderLayer(d_model=FLAGS.model_size, nhead=2, relative_positional=True, relative_positional_distance=100, dim_feedforward=3072, dropout=FLAGS.dropout)
        #self.transformer = nn.TransformerEncoder(encoder_layer, FLAGS.num_layers)
        self.fc = nn.Linear(FLAGS.model_size*75, num_classes) #mfcc prediction
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.25)
        #self.fc = nn.Linear(num_outs, num_classes)



    def forward(self, x_raw):
        # x shape is (batch, time, electrode)
        x = x_raw
        x_raw = x.transpose(1,2) # put channel before time for conv so that the conv can be alonge chanl
        x_raw = self.conv_blocks(x_raw)
        x = self.fc(self.dropout(self.flatten(x_raw)))
        #x = x_raw.transpose(1,2)

        return x

def save_output(model, datapoint, filename, device, audio_normalizer, vocoder):
    model.eval()
    with torch.no_grad():
        sess = torch.tensor(datapoint['session_ids'], device=device).unsqueeze(0)
        X = torch.tensor(datapoint['emg'], dtype=torch.float32, device=device).unsqueeze(0)
        X_raw = torch.tensor(datapoint['raw_emg'], dtype=torch.float32, device=device).unsqueeze(0)

        pred, _ = model(X, X_raw, sess)
        y = pred.squeeze(0)

        y = audio_normalizer.inverse(y.cpu()).to(device)

        audio = vocoder(y).cpu().numpy()

    sf.write(filename, audio, 22050)

    model.train()

def test(model, testset, device):
    model.eval()

    dataloader = torch.utils.data.DataLoader(testset, batch_size=4)
    losses = []
    accuracies = []
    correct =0
    criterion = nn.CrossEntropyLoss()
    grnd_all = []
    predctd_all = []
    with torch.no_grad():
        for batch in dataloader:
            label, data = batch
            X = data.to(device, non_blocking=True)
            label = label.type(torch.LongTensor)
            lbl = label.to(device,non_blocking=True)
            pred1 = model(X)
            loss = criterion(pred1, lbl)
            losses.append(loss.item())
            pred= torch.argmax(pred1, dim=1)
            predctd_all.append(pred.cpu().numpy())
            grnd_all.append(lbl.cpu().numpy())
            correct = pred.eq(lbl.view_as(pred)).sum().item()
            accuracy = correct/4
            accuracies.append(accuracy)

    predctd_all = [item for tnsr in predctd_all for item in tnsr]
    grnd_all = [item for tnsr in grnd_all for item in tnsr] #[item for item in grnd_all]
    f1 = f1_score(grnd_all, predctd_all)
    precision= precision_score(grnd_all, predctd_all)
    recall = recall_score(grnd_all, predctd_all)
    Acc_mean = np.mean(accuracies)
    Loss_ = np.mean(losses)
    logging.info(f'Mean Accuracy {Acc_mean} - percision score: {precision:.4f} Recall score: {recall:.4f} F1 score: {f1*100:.2f} mean Loss: {Loss_*100:.2f} ')

    model.train()
    return np.mean(accuracies),np.mean(losses)
def train_model(trainset, devset, device):
    n_epochs = FLAGS.epochs
    Signal_dimention = 6#25#6
    Num_classes = 2
    save_sound_outputs = True

    if FLAGS.data_size_fraction >= 1:
        training_subset = trainset
    else:
        training_subset = trainset.subset(FLAGS.data_size_fraction)#, pin_memory=(device=='cpu')  'cuda'  256000

    model = Model(Signal_dimention, Num_classes).to(device)

    dataloader = torch.utils.data.DataLoader(training_subset, num_workers=4,
                                             batch_size=256)  # batch_sampler=SizeAwareSampler(training_subset, 22050)) #

    if FLAGS.start_training_from is not None:
        state_dict = torch.load(FLAGS.start_training_from)
        model.load_state_dict(state_dict, strict=False)

    #if save_sound_outputs:
    #    vocoder = Vocoder()

    optim = torch.optim.Adam(model.parameters(), weight_decay=FLAGS.l2)
    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', 0.5, patience=FLAGS.learning_rate_patience)

    def set_lr(new_lr):
        for param_group in optim.param_groups:
            param_group['lr'] = new_lr

    target_lr = FLAGS.learning_rate
    def schedule_lr(iteration):
        iteration = iteration + 1
        if iteration <= FLAGS.learning_rate_warmup:
            set_lr(iteration*target_lr/FLAGS.learning_rate_warmup)

    seq_len = 300#200 # sequence length of emg features where emg raw seq_len*4 channel

    batch_idx = 0
    criterion = nn.CrossEntropyLoss().to(device)  # (applies Softmax)
    for epoch_idx in range(n_epochs):
        losses = []
        for batch in dataloader:  #
        #for j in range(1000):
            optim.zero_grad()
            schedule_lr(batch_idx)
            label, data = batch
            X = data.to(device, non_blocking=True)
            label = label.type(torch.LongTensor)
            lbl = label.to(device,non_blocking=True)
            y_pred = model(X)
            loss = criterion(y_pred, lbl)
            loss.backward()
            optim.step()

            losses.append(loss.item())


            print('loss', loss.item())

            print('the batch_index is =', batch_idx)
            if batch_idx%20==0:
                #if epoch_idx%2==0:
                train_loss = np.mean(losses)
                val_acc, val_loss = test(model, devset, device)
                lr_sched.step(val_acc)
                logging.info(f'finished epoch {epoch_idx+1} - validation loss: {val_loss:.4f} training loss: {train_loss:.4f} class accuracy: {val_acc*100:.2f}')
            batch_idx += 1
        print('Epoch idx', epoch_idx)
        torch.save(model.state_dict(), os.path.join(FLAGS.output_directory, 'model.pt'))


    return model

def main():
    BaseDirectory = '../Filtered Data/'
    train_idx = [0,1,3,4,5,7,8,9,10,11,12]  # 0 best
    test_idx = [2]  # 2,6 90 percent accuracy F1 94
    tasks_types = [0,1,2,3]
    os.makedirs(FLAGS.output_directory, exist_ok=True)
    logging.basicConfig(handlers=[
            logging.FileHandler(os.path.join(FLAGS.output_directory, 'log.txt'), 'w'),
            logging.StreamHandler()
            ], level=logging.INFO, format="%(message)s")

    logging.info(sys.argv)


    trainset= MMFoGDataset(BaseDirectory,train_idx,tasks_types)
    testset = MMFoGDataset(BaseDirectory,test_idx,tasks_types)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = train_model(trainset, testset, device)

if __name__ == '__main__':
    FLAGS(sys.argv)
    main()
