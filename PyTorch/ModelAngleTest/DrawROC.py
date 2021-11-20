'''
本文件用于ROC曲线绘制以及AUC计算
'''

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

from itertools import cycle
from sklearn import svm,datasets
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from sklearn.metrics import  roc_curve,auc
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier


Work_Path = "F:\\PlantarPressurePredictExperiment"
os.chdir(Work_Path)


#region ModelImport
#选择网络
NetWorkChoose = "LSTM"

if(NetWorkChoose == "LSTM"):
    #LSTM
    from ProcessProgram.PyTorch.LSTM import LSTM
    from ProcessProgram.PyTorch.LSTM import sequenceLen
    from ProcessProgram.PyTorch.LSTM import getmodel,BATCH_SIZE
elif(NetWorkChoose == "CNN"):
    #CNN
    from ProcessProgram.PyTorch.CNN import CNN
    from ProcessProgram.PyTorch.CNN import getmodel,BATCH_SIZE

elif(NetWorkChoose == "ResNet18"):
    # ResNet18
    from ProcessProgram.PyTorch.ResNet18 import *
    from ProcessProgram.PyTorch.ResNet18 import getmodel,BATCH_SIZE
#endregion

global x_test,y_test,test_loader
global rocArray
rocArray = np.array([0,0,0,0,0])

#region 文件导入
def importData():
    SaveTestDataFilePath = "Pytorch\\data\\angle\\TestData.csv"
    SaveTestLabelFilePath = "Pytorch\\data\\angle\\TestLabel.csv"
    global x_test,y_test
    with open(SaveTestDataFilePath, encoding='utf-8') as path_x_test_m, \
        open(SaveTestLabelFilePath, encoding='utf-8')as path_y_test_m:

        x_test = np.loadtxt(path_x_test_m, dtype=float,delimiter=",")
        y_test = np.loadtxt(path_y_test_m,dtype = int, delimiter=",")
#endregion

#region  数据转换
def TrainFormData():
    global x_test,y_test,test_loader

    if (NetWorkChoose == "LSTM"):
        x_test = x_test.reshape(len(x_test), sequenceLen, 1260)
        # CNN /Resnet RESHAPE
    else:
        x_test = x_test.reshape(len(x_test), 1, 60, 21)

    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)

    x_test =  torch.tensor(x_test,dtype= torch.float32)
    y_test = torch.tensor(y_test,dtype=torch.long)

    test_dataset = Data.TensorDataset(x_test,y_test)
    test_loader = Data.dataloader.DataLoader(dataset= test_dataset,batch_size = BATCH_SIZE ,shuffle= True)
#endregion


def TestNetWork(model):
    global rocArray
    correct = 0
    correctCount = [0,0,0,0,0]      #0，30，60，90  正确数量
    TotalCount = [0,0,0,0,0]        #总数
    ErrRect = np.zeros((5,5))
    for step, (data, target) in enumerate(test_loader):
        data, target = Variable(data).cuda(),Variable(target).cuda()
        output = model(data)
        # sum up batch loss
        # get the index of the max log-probability
        # ans = F.softmax(output,dim=1)       #
        ans = output


        for sample in range(len(ans)):
            line = [int(target[sample]),float(ans[sample][0]),float(ans[sample][1]),float(ans[sample][2]),float(ans[sample][3])]
            rocArray = np.row_stack((rocArray,np.array(line)))


        pred = torch.max(output.data, 1)[1]
        correct += int(pred.eq(target.data.view_as(pred)).cpu().sum())

        for ansIndex in range(len(pred)):
            angleIndex = target[ansIndex]     #真值
            predictIndex = pred[ansIndex]     #预测值
            if(predictIndex == angleIndex):
                correctCount[angleIndex] += 1
                TotalCount[angleIndex] += 1
            else:
                TotalCount[angleIndex] += 1
                ErrRect[angleIndex][pred[ansIndex]] +=1


    correctRate = 100. * correct / len(test_loader.dataset)
    print('Accuracy: {}/{} ({:.3f}%)\n'.format(correct, len(test_loader.dataset),correctRate))
    print("Count：0°:{}/{} 30°:{}/{}  60°:{}/{}  90°:{}/{}".format(correctCount[0],TotalCount[0],
                                                                  correctCount[1],TotalCount[1],
                                                                  correctCount[2],TotalCount[2],
                                                                  correctCount[3],TotalCount[3],))
    print("angelAccuracy：0°:{:.2f}%  30°:{:.2f}%  60°:{:.2f}%  90°:{:.2f}% ".format(100. * correctCount[0]/TotalCount[0],
                                                                                   100. * correctCount[1]/TotalCount[1],
                                                                                   100. * correctCount[2]/TotalCount[2],
                                                                                   100. * correctCount[3]/TotalCount[3],))
    print("错误矩阵:-------------\n",ErrRect)



    return ("{:.3f}%".format(100. * correct / len(test_loader.dataset)))


#region ROC曲线绘制

def CalculateROC():
    target = rocArray[:,0]    #真实标签集
    y_score = rocArray[:,1:5] #得分集

    target = label_binarize(target, classes=[0, 1, 2, 3])
    n_classes = 4

    fpr = {}
    tpr = {}
    roc_auc ={}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(target[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area   计算微平均。
    fpr["micro"], tpr["micro"], _ = roc_curve(target.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    return fpr,tpr,roc_auc

def DrawRocAndAUC(fpr,tpr,roc_auc):
    #这个是画单个角度的。
    plt.figure()
    lw = 2
    plt.plot(
        fpr[1],
        tpr[1],
        color="darkorange",
        lw=lw,
        label="0° ROC 曲线 (area = %0.2f)" % roc_auc[1],
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic ")
    plt.legend(loc="lower right")
    plt.show()
    pass

def DrawRocAndAUCAll(fpr,tpr,roc_auc):
    n_classes = 4
    lw = 2
    angle = ["0°","30°","60°","90°"]
    # 多角度分类全部绘制 绘制所有的假阳性率，全部保持唯一
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    #插值计算，用于画出汇总曲线
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    #插值加起来了需要除掉
    mean_tpr /= n_classes

    #计算宏平均
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    #画出所有ROC曲线
    plt.figure()

    #中文展示
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    #微平均
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="微平均 ROC 曲线 (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )
    #宏平均
    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="宏平均 ROC 曲线 (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue","green"])

    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="ROC 曲线 {0} (area = {1:0.2f})".format(angle[i], roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("假阳性")
    plt.ylabel("真阳性")
    plt.title("多角度分类接收者操作特征曲线")
    plt.legend(loc="lower right")
    plt.show()

#endregion




if __name__ == '__main__':
    importData()
    TrainFormData()
    #resNetET10_25_1_correct88%.pkl
    #lstm10_8_1_correct81%.pkl
    model = getmodel("lstm10_8_1_correct81%.pkl")        #放入模型
    TestNetWork(model)
    #在这里已经完成rocArray 目标预测矩阵  [真值,0°预测概率,30°预测速率,60°预测概率,90°预测概率]
    #接下来进行ROC绘制以及AUC计算

    fpr,tpr,roc_auc = CalculateROC()
    DrawRocAndAUCAll(fpr,tpr,roc_auc)

    pass