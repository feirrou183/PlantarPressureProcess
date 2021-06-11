import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


Work_Path = "F:\\PlantarPressurePredictExperiment"
os.chdir(Work_Path)


def getDic():
    with open("subjectConfig.json", "r",encoding= "utf-8") as f:
        subjectConfigDict = json.load(f)
    return subjectConfigDict





def GetKeyPoint(dic,subject,item,step):
    HC = dic[subject]["ResultData"][item][step].split(",")[0]
    SC = dic[subject]["ResultData"][item]["{0}SCValue".format(step)].split("_")[0]
    HL = dic[subject]["ResultData"][item]["{0}HLValue".format(step)].split("_")[0]
    TO = dic[subject]["ResultData"][item][step].split(",")[1]
    return int(HC),int(SC),int(HL),int(TO)


def GetVelocityArr(subject,detailItemName,eachStep,HC,SC,HL,TO):
    file = "1"
    if(eachStep == "BTS" or eachStep == "TS"):
        file = open("ProcessedData\\subject{0}\\{1}-L.csv".format(subject,detailItemName),encoding="utf-8")
    elif(eachStep == "BAP" or eachStep == "AP" or eachStep == "DS"):
        file = open("ProcessedData\\subject{0}\\{1}-R.csv".format(subject,detailItemName),encoding="utf-8")
    else:
        print("Error",subject,detailItemName)

    Arr = np.loadtxt(file,delimiter = ",")
    HCVelocityArr = Arr[HC-1:SC-1,:] - Arr[HC-2:SC-2,:]
    MSVelocityArr = Arr[SC:HL-1,:] - Arr[SC - 1:HL-2,:]
    TOVelocityArr = Arr[HL:TO-1,:] - Arr[HL - 1:TO- 2,:]
    file.close()
    return HCVelocityArr,MSVelocityArr,TOVelocityArr


def GetLabel(dic,subjectName,detailItemName):
    angle = dic[subjectName]["ResultData"][detailItemName]["angle"]
    strategy = dic[subjectName]["ResultData"][detailItemName]["strategy"]
    return angle,strategy




def GetFileIterator(dic):
    for subject in subjects:
        for item in items:
            for sub_item in sub_items:
                detailItemName = "{0}-{1}".format(item, sub_item)
                subjectName = "subject{}".format(subject)
                if (dic[subjectName]["ResultData"].__contains__(detailItemName)):
                    for eachStep in Step:
                        HC, SC, HL, TO = GetKeyPoint(dic, subjectName, detailItemName, eachStep)
                        HCVelocityArr, MSVelocityArr, TOVelocityArr = GetVelocityArr(subject, detailItemName, eachStep, HC, SC, HL, TO)
                        angle,strategy =  GetLabel(dic,subjectName,detailItemName)
                        #if(angle != "0°") : continue      #不要30°和120°变化
                        for k in range(len(TOVelocityArr)):
                            #yield subjectName,detailItemName,angle,strategy,eachStep,HCArr,MSArr,TOArr
                            yield subjectName, detailItemName, angle, strategy, eachStep,TOVelocityArr[k]
                        print(subjectName,"---",detailItemName)


def switchLabelClass(label):
    if label == 0:
        return 0
    if label ==30:
        return 1
    if label ==60:
        return 2
    if label == 90:
        return 3
    if label == 120:
        return 4

#标准化
def normalization(Arr):
    mean = Arr.mean()
    std = Arr.std()
    Arr = (Arr - mean) / std  # 标准化数据
    return Arr



if __name__ == '__main__':
    subjects = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    items = ["1", "2", "3", "4", "5"]
    sub_items = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
    #Step = ["BAP", "BTS", "AP", "TS", "DS"]
    Step = ["AP"]
    SaveTrainDataFilePath = "Pytorch\\data\\angle\\TrainData.csv"
    SaveTrainLabelFilePath = "Pytorch\\data\\angle\\TrainLabel.csv"
    SaveTestDataFilePath = "Pytorch\\data\\angle\\TestData.csv"
    SaveTestLabelFilePath = "Pytorch\\data\\angle\\TestLabel.csv"

    dic = getDic()
    TrainArr = []
    TrainAngle = []
    TestArr = []
    TestAngle = []

#region   图像展示
    '''
    fig = plt.figure()
    plt.ion()
    f = GetFileIterator(dic)
    for eachItem in GetFileIterator(dic):
        a = f.__next__()
        Arr = normalization(a[5])
        Arr = np.reshape(Arr,(60,21))
        plt.imshow(Arr,cmap="Greys")
        plt.title(a[1])
        plt.pause(0.02)
    plt.ioff()
    
    '''
#endregion

#region 图像输出
    i= 0
    for eachItem in GetFileIterator(dic):
        Arr = eachItem[5]
        label = int(eachItem[2].split("°")[0])
        label = switchLabelClass(label)
        Arr = normalization(Arr)  #数据标准化
        i +=1
        if(i % 5 != 0):
            TrainArr.append(Arr)
            TrainAngle.append(label)
        else:
            TestArr.append(Arr)
            TestAngle.append(label)

    TrainArr = np.array(TrainArr)
    TrainAngle = np.array(TrainAngle)
    TestArr = np.array(TestArr)
    TestAngle = np.array(TestAngle)

    np.savetxt(SaveTrainDataFilePath,TrainArr,delimiter=",")
    np.savetxt(SaveTrainLabelFilePath, TrainAngle,fmt="%d", delimiter=",")

    np.savetxt(SaveTestDataFilePath, TestArr, delimiter=",")
    np.savetxt(SaveTestLabelFilePath, TestAngle,fmt="%d", delimiter=",")
#endregion






























