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

# # ResNet18
from ProcessProgram.PyTorch.ResNet18 import *
from ProcessProgram.PyTorch.ResNet18 import getmodel,BATCH_SIZE

global x_test,y_test,test_loader

BATCH_SIZE = 6  #一次进入的个数
def importData():
    SaveTestDataFilePath = "Pytorch\\data\\angle\\TestData.csv"
    SaveTestLabelFilePath = "Pytorch\\data\\angle\\TestLabel.csv"
    global x_test,y_test
    with open(SaveTestDataFilePath, encoding='utf-8') as path_x_test_m, \
        open(SaveTestLabelFilePath, encoding='utf-8')as path_y_test_m:

        x_test = np.loadtxt(path_x_test_m, dtype=float,delimiter=",")
        y_test = np.loadtxt(path_y_test_m,dtype = int, delimiter=",")

def TrainFormData():
    global x_test,y_test,test_loader
    #LSTM RESHAPE
    # x_test = x_test.reshape(len(x_test),sequenceLen,1260)
    #CNN /Resnet RESHAPE
    x_test = x_test.reshape(len(x_test),1,60,21)

    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)

    x_test = torch.tensor(x_test,dtype= torch.float32)
    y_test = torch.tensor(y_test,dtype=torch.long)

    test_dataset = Data.TensorDataset(x_test,y_test)
    test_loader = Data.dataloader.DataLoader(dataset= test_dataset,batch_size = BATCH_SIZE ,shuffle= False)    #shuffle= False 不打乱次序

def useCoreAnalyse(pred,target):
    angle = [0,0,0,0]
    targetAngle = [0,0,0,0]
    try:
        for i in range(BATCH_SIZE):
            predIndex = int(pred[i])
            predTargetIndex = int(target[i])
            if(predIndex == predTargetIndex):
                angle[predIndex] +=1
            targetAngle[predTargetIndex] +=1
        predAngle = angle.index(max(angle))
        realAngle = targetAngle.index((max(targetAngle)))
        if(predAngle == realAngle):
            return True
        return False
    except IndexError:
        return False






def TestNetWork(model):
    correct = 0
    correctCount = [0,0,0,0,0]      #0，30，60，90  正确数量
    TotalCount = [0,0,0,0,0]        #总数
    ErrRect = np.zeros((5,5))
    tempList = []  #中间数组
    for step, (data, target) in enumerate(test_loader):
        data, target = Variable(data).cuda(),Variable(target).cuda()
        output = model(data)
        # sum up batch loss
        # get the index of the max log-probability
        pred = torch.max(output.data, 1)[1]
        if(useCoreAnalyse(pred,target)):
            correct+=1

    correctRate = 100. * correct / (len(test_loader.dataset)//BATCH_SIZE)
    print("{:.0f}%".format(correctRate))


    return ("{:.0f}%".format(100. * correct / (len(test_loader.dataset)//BATCH_SIZE)))


if __name__ == '__main__':
    importData()
    TrainFormData()
    model = getmodel("resNetET10_25_1_correct88%.pkl")        #放入模型
    TestNetWork(model)

    pass