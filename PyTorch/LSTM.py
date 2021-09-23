import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy
import torch.utils.data as Data
import torch
import sys
import os
from ProcessProgram.NeurNetWorkProcess.ProceRawDataToTensor import *


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
        self.lstm1 = nn.LSTM(1260,40,3,batch_first=True)          #输入特征1260，输出特征50, 3层隐藏层
        self.out = nn.Linear(40,4)

    #前向函数
    def forward(self, x):
        #x = self.batchNorm(x)
        x = self.lstm1(x)[0]
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


def TestNetWork(lstm):
    lstm.eval()
    correct = 0
    test_loss = 0
    for step, (data, target) in enumerate(test_loader):
        data, target = Variable(data),Variable(target)
        data = data.float()
        output = lstm(data)
        # sum up batch loss
        test_loss += loss_func(output, target).item()
        # get the index of the max log-probability
        pred = torch.max(output.data, 1)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        test_loss /= len(test_loader.dataset)
    correctRate = int(100. * correct / len(test_loader.dataset))
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),correctRate))
    tempMax = 85
    if(correctRate > tempMax):
        tempMax = correctRate
        #savemodel(cnn, "tempMaxModel\\cnn_tempMax_correct{}%.pkl".format(correctRate))
    lstm.train()
    return ("{:.0f}%".format(100. * correct / len(test_loader.dataset)))



if __name__ == '__main__':
    importData()
    TrainFormData()
    lstm = LSTM()
    lstm.cuda()

    optimizer = torch.optim.Adam(lstm.parameters(), lr=Learn_Rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5,
    #                                             last_epoch=-1)  # 动态调整学习率。每10轮下降一位小数点
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(1,EPOCH):
        for step,(x,y) in enumerate(train_loader):
            b_x = Variable(x)
            b_y = Variable(y)
            output = lstm(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step % 500 == 0):
                print('Train Epoch: {} \t [{:4d}/{:4d} ({:.0f}%)] \t\t Loss: {:.6f}'.format(
                    epoch, step * len(x), len(train_loader.dataset),
                           100. * step / len(train_loader), loss.data), end="\n")
        #scheduler.step()

    print("Final:", end="")
    # test
    Correct = TestNetWork(lstm)
    savemodel(lstm, "cnn9_23_0_correct{}.pkl".format(Correct))


