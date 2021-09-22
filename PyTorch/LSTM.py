import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy
import torch.utils.data as Data
import torch
import sys
import os
import numpy as np

Work_Path = "F:\\PlantarPressurePredictExperiment"
os.chdir(Work_Path)
global x_train,x_test,y_train,y_test,train_loader,test_loader

#region 预置参数
BATCH_SIZE = 8
Learn_Rate = 0.001
EPOCH = 30
tempMax = 87

sequenceLen = 2         #视频序列长度
#endregion

#region 文件导入
def importData():
    SaveTrainDataFilePath = "Pytorch\\data\\angle\\TrainData.csv"
    SaveTrainLabelFilePath = "Pytorch\\data\\angle\\TrainLabel.csv"
    SaveTestDataFilePath = "Pytorch\\data\\angle\\TestData.csv"
    SaveTestLabelFilePath = "Pytorch\\data\\angle\\TestLabel.csv"
    global x_train,x_test,y_train,y_test
    with open(SaveTrainDataFilePath, encoding='utf-8') as path_x_train_m, \
            open(SaveTestDataFilePath, encoding='utf-8') as path_x_test_m, \
            open(SaveTrainLabelFilePath, encoding='utf-8') as path_y_train_m, \
            open(SaveTestLabelFilePath, encoding='utf-8')as path_y_test_m:

        x_train = np.loadtxt(path_x_train_m,dtype=float,delimiter=",")
        x_test = np.loadtxt(path_x_test_m, dtype=float,delimiter=",")

        y_train = np.loadtxt(path_y_train_m,dtype= int,delimiter=",")
        y_test = np.loadtxt(path_y_test_m,dtype = int, delimiter=",")
#endregion

#region  数据转换
def TrainFormData():
    global x_train,x_test,y_train,y_test,train_loader,test_loader
    x_train = x_train.reshape(len(x_train),sequenceLen,1260)
    x_test = x_test.reshape(len(x_test),sequenceLen,1260)

    x_train = torch.from_numpy(x_train)
    x_test = torch.from_numpy(x_test)

    y_train = torch.from_numpy(y_train)
    y_test = torch.from_numpy(y_test)

    x_train = torch.tensor(x_train,dtype= torch.float32)
    x_test =  torch.tensor(x_test,dtype= torch.float32)

    y_train = torch.tensor(y_train,dtype=torch.long)
    y_test = torch.tensor(y_test,dtype=torch.long)

    train_dataset = Data.TensorDataset(x_train,y_train)
    test_dataset = Data.TensorDataset(x_test,y_test)

    train_loader = Data.dataloader.DataLoader(dataset= train_dataset,batch_size = BATCH_SIZE ,shuffle= True)
    test_loader = Data.dataloader.DataLoader(dataset= test_dataset,batch_size = BATCH_SIZE ,shuffle= True)
#endregion

#region lstm模型
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM,self).__init__()
        self.batchNorm =nn.BatchNorm2d(1260, momentum=0.1)
        self.lstm1 = nn.LSTM(1260,50,3,batch_first=True)          #输入特征1260，输出特征50, 3层隐藏层
        self.out = nn.Linear(50,5)

    #前向函数
    def forward(self, x):
        #x = self.batchNorm(x)
        x = self.lstm1(x)
        x = x.view(x.size(0), -1)  #将除Batch的维度展成一维的
        output = self.out(x)
#endregion

#region  功能函数

#保存网络
def savemodel(model,filename_Date_correctRate):
    torch.save(model,"ProcessProgram\\model\\{}".format(filename_Date_correctRate))

#提取网络
def getmodel(filename_Date_correctRate):
    net = torch.load("ProcessProgram\\model\\{}".format(filename_Date_correctRate))
    return net
#endregion


if __name__ == '__main__':
    importData()
    TrainFormData()

