import torch
import torch.nn as nn
from torch.nn import functional as F
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

#region 预置参数
BATCH_SIZE = 8
Learn_Rate = 0.001
EPOCH = 30
tempMax = 87
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

#region  resNet18
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
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
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
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        return out
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

#region  功能函数

#保存网络
def savemodel(model,filename_Date_correctRate):
    torch.save(model,"Pytorch\\model\\{}".format(filename_Date_correctRate))

#提取网络
def getmodel(filename_Date_correctRate):
    net = torch.load("Pytorch\\model\\{}".format(filename_Date_correctRate))
    return net
#endregion

if __name__ == '__main__':
    importData()
    TrainFormData()
    cnn = RestNet18()
    cnn.cuda()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=Learn_Rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5,
                                                last_epoch=-1)  # 动态调整学习率。每10轮下降一位小数点
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCH):
        for step, (x, y) in enumerate(train_loader):
            b_x = Variable(x)
            b_y = Variable(y)
            output = cnn(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step % 500 == 0):
                print('Train Epoch: {} \t [{:4d}/{:4d} ({:.0f}%)] \t\t Loss: {:.6f}'.format(
                    epoch, step * len(x), len(train_loader.dataset),
                           100. * step / len(train_loader), loss.data), end="\n")
                TestNetWork(cnn)
                # print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy() , end= "")
        scheduler.step()

    print("Final:", end="")
    # test
    Correct = TestNetWork(cnn)

    savemodel(cnn, "resNetEt10_12_0_correct{}.pkl".format(Correct))
