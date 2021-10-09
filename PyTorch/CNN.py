import numpy as np
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

Work_Path = "F:\\PlantarPressurePredictExperiment"
os.chdir(Work_Path)
global x_train,x_test,y_train,y_test,train_loader,test_loader
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

#region 预置参数
BATCH_SIZE = 16
Learn_Rate = 0.001
EPOCH = 60
tempMax = 87
#endregion

#region  数据转换
def TrainFormData():
    global x_train,x_test,y_train,y_test,train_loader,test_loader
    x_train = x_train.reshape(len(x_train),1,60,21)
    x_test = x_test.reshape(len(x_test),1,60,21)

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

#region CNN网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        #region  conv1
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding = 2,  # padding = (kernel_size -1)/2
                ),
        nn.ReLU(),
        #nn.AvgPool2d(2)
        )
        #endregion
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(64, momentum=0.3),
            nn.Conv2d(64, 128, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(128, momentum=0.1),
            nn.Conv2d(128,256,3,1,1),
            nn.Softmax2d(),
            nn.MaxPool2d(2),
        )

        self.conv4 = nn.Sequential(
            nn.Dropout(0.5),
            nn.BatchNorm2d(256,momentum=0.1),
            nn.Conv2d(256,512,3,1,2),
            nn.Softmax2d(),
            nn.MaxPool2d(2)
        )


        self.out = nn.Linear(256 * 15 * 5, 5)  #5分类

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)  # batch,32,15,5
        x = self.conv3(x)  #  256 * 15 * 5
        #x = self.conv4(x)  #  128 * 8 * 3
        x = x.view(x.size(0), -1)  # (batch,32*15*5)
        output = self.out(x)
        return output
#endregion


#region  功能函数

#保存网络
def savemodel(model,filename_Date_correctRate):
    torch.save(model,"Pytorch\\model\\{}".format(filename_Date_correctRate))

#提取网络
def getmodel(filename_Date_correctRate):
    net = torch.load("Pytorch\\model\\{}".format(filename_Date_correctRate))
    return net
#endregion

def TestNetWork(cnn):
    cnn.eval()
    correct = 0
    test_loss = 0
    for step, (data, target) in enumerate(test_loader):
        data, target = Variable(data).cuda(),Variable(target).cuda()
        data = data.float()
        output = cnn(data)
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
    cnn.train()
    return ("{:.0f}%".format(100. * correct / len(test_loader.dataset)))

def getAccuracyOfTrainging():
    pass



if __name__ == '__main__':
    importData()
    TrainFormData()
    cnn = CNN()
    optimizer = torch.optim.Adam(cnn.parameters(),lr = Learn_Rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size= 20,gamma=0.5,last_epoch= -1)  #动态调整学习率。每10轮下降一位小数点
    loss_func = nn.CrossEntropyLoss()

    print("使用GPU:", torch.cuda.get_device_name(0))
    cnn = cnn.cuda()
    loss_func = loss_func.cuda()




    for epoch in range(1,EPOCH):
        for step,(x,y) in enumerate(train_loader):
            b_x = Variable(x).cuda()
            b_y = Variable(y).cuda()
            output = cnn(b_x)
            loss = loss_func(output,b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            if(step % 500 == 0):
                print('Train Epoch: {} \t [{:4d}/{:4d} ({:.0f}%)] \t\t Loss: {:.6f}'.format(
                    epoch, step * len(x), len(train_loader.dataset),
                           100. * step / len(train_loader),loss.data),end= "\n")
                #TestNetWork(cnn)
                #print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy() , end= "")
            if(step % 10000 == 0):
                TestNetWork(cnn)
        scheduler.step()

    print("Final:" , end= "")
    # test
    Correct = TestNetWork(cnn)

    savemodel(cnn,"cnn10_9_1_correct{}.pkl".format(Correct))








