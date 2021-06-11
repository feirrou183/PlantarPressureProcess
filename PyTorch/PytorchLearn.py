import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy
import torch.utils.data as Data
import torch
import sys



EPOCH = 1  #训练次数
BATCH_SIZE =  None
TIME_STEP = None
INPUT_SIZE = None
HIDDEN_SIZE = None
OUTPUT_SIZE = None
LearnRate = 0.01

train_dataset = []

train_data = None

test_data = None
test_x = Variable(test_data.val)
test_y = Variable(test_data.label)

train_loader = Data.dataloader.DataLoader(dataset= train_dataset,batch_size = BATCH_SIZE ,shuffle= True)


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size= HIDDEN_SIZE,
            num_layers= 1 ,                 #1层实验

        )
        self.out = nn.Linear(HIDDEN_SIZE,OUTPUT_SIZE)

    def forward(self,x,h_state):
        out,h_state = self.lstm(x,h_state)
        return out,h_state

optimizer = optim.Adam(LSTM.parameters(),lr=LearnRate)
loss_func = nn.CrossEntropyLoss()

h_state = None
for epoch in range(EPOCH):
    for step,(x,y) in enumerate(train_loader):
        b_x = Variable(x.view(-1,TIME_STEP,INPUT_SIZE))
        b_y = Variable(y)

        prediction,h_state = LSTM(b_x,h_state)
        h_state = Variable(h_state.data)

        loss = loss_func(prediction,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

















