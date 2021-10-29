import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
import torch.utils.data as Data
import torch
import sys
import os

import LSTM
import ResNet18

Work_Path = "F:\\PlantarPressurePredictExperiment"
os.chdir(Work_Path)
global x_train,x_test,y_train,y_test,train_loader,test_loader

#region 预置参数
BATCH_SIZE = 16
Learn_Rate = 0.001
EPOCH = 60           #可以考虑收敛算法，计算出不再增长后跳出训练(20轮训练不增长)
tempMax = 87
sequenceLen = 6         #视频序列长度
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
    print("模型转换...")
    global x_train,x_test,y_train,y_test,train_loader,test_loader
    x_train = x_train.reshape(len(x_train),sequenceLen,1,60,21)
    x_test = x_test.reshape(len(x_test),sequenceLen,1,60,21)

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


#region resNet18
class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x + output)

class RestNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))

        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(extra_x + out)

class RestNet18(nn.Module):
    def __init__(self):
        super(RestNet18, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(RestNetBasicBlock(64, 64, 1),
                                    RestNetBasicBlock(64, 64, 1))
        self.layer2 = nn.Sequential(RestNetDownBlock(64, 128, [2, 1]),
                                    RestNetBasicBlock(128, 128, 1))
        self.layer3 = nn.Sequential(RestNetDownBlock(128, 256, [2, 1]),
                                    RestNetBasicBlock(256, 256, 1))
        self.layer4 = nn.Sequential(RestNetDownBlock(256, 512, [2, 1]),
                                    RestNetBasicBlock(512, 512, 1))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512, 4)


    def forward(self, x):

        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        return out.squeeze()

#endregion


#region 融合模型
class MultiModule(nn.Module):
    def __init__(self):
        super(MultiModule, self).__init__()
        self.resNet18 = RestNet18()
        self.lstm = LSTM.LSTM(512,1260,1,4)
        self.FeatureArray = np.zeros((sequenceLen,512))

    def resNetCalculate(self,x):
        out0 = self.resNet18(x[:, 0])
        out1 = self.resNet18(x[:, 1])
        out2 = self.resNet18(x[:, 2])
        out3 = self.resNet18(x[:, 3])
        out4 = self.resNet18(x[:, 4])
        out5 = self.resNet18(x[:, 5])
        out = torch.stack([out0, out1, out2, out3, out4, out5], dim=1)
        return out



    def forward(self,x):
        out = self.resNetCalculate(x)
        out = self.lstm(out)
        return out
#endregion

def TestNetWork(lstm):
    lstm.eval()
    correct = 0
    test_loss = 0
    tempMax = 80
    for step, (data, target) in enumerate(test_loader):
        data, target = Variable(data).cuda(),Variable(target).cuda()
        output = lstm(data)
        # sum up batch loss
        test_loss += loss_func(output, target).item()
        # get the index of the max log-probability
        pred = torch.max(output.data, 1)[1]
        correct += int(pred.eq(target.data.view_as(pred)).cpu().sum())
        test_loss /= len(test_loader.dataset)
    correctRate = round(100. * correct / len(test_loader.dataset),2)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),correctRate))
    if(correctRate > tempMax):
         tempMax = correctRate
         LSTM.savemodel(lstm, "tempMaxModel\\lstm_tempMax_correct{}%.pkl".format(correctRate))
    lstm.train()
    return ("{:.0f}%".format(100. * correct / len(test_loader.dataset)))



if __name__ == '__main__':

    importData()
    TrainFormData()
    multiModule = MultiModule()
    multiModule.cuda()

    optimizer = torch.optim.Adam(multiModule.parameters(), lr=Learn_Rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5,
                                                last_epoch=-1)  # 动态调整学习率。每10轮下降一位小数点
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        for step,(x,y) in enumerate(train_loader):
            b_x = Variable(x).cuda()
            b_y = Variable(y).cuda()
            output = multiModule(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step % 500 == 0):
                print('Train Epoch: {} \t [{:4d}/{:4d} ({:.0f}%)] \t\t Loss: {:.6f}'.format(
                    epoch, step * len(x), len(train_loader.dataset),
                           100. * step / len(train_loader), loss.data), end="\n")
                TestNetWork(multiModule)
        scheduler.step()

    print("Final:", end="")
    # test
    Correct = TestNetWork(multiModule)
    LSTM.savemodel(multiModule, "resPlusLstm10_27_0_correct{}.pkl".format(Correct))



