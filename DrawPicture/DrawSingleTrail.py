import os
import matplotlib.pyplot as plt
import numpy as np
import csv
import ProcessProgram.Velocity.PostionVelocityDataPreProcess as dicManager
from matplotlib import cm
import cv2 as cv


# 设置工作环境
Work_Path = "F:\\PlantarPressurePredictExperiment"
os.chdir(Work_Path)

#subject 03
#60°  683-695   1-1
#90°  585-597   2-4
#0°   1250-1262   4-1
#30°---> Error

def drawTrail(subject,detailItemName,Index,End):
    start = Index
    end = End
    plt.style.use("ggplot")
    cmap = cm.get_cmap('jet')
    plt.rcParams["axes.grid"] = False
    HLArr = GetArr(subject, detailItemName, start, end)
    plt.figure()

    for i in range(end-start):
        picture = HLArr[i].reshape(60,21)
        plt.subplot(1,end-start,i+1)
        plt.imshow(picture,cmap=cmap)
        plt.axis("off")
        plt.xticks([])
        plt.yticks([])

    plt.show()







def GetArr(subject,detailItemName,HL,TO):

    file = open("ProcessedData\\{0}\\{1}-R.csv".format(subject,detailItemName),encoding="utf-8")
    Arr = np.loadtxt(file,delimiter = ",")

    TOArr = Arr[HL:TO,:]
    file.close()

    return TOArr


if __name__ == '__main__':
    drawTrail("subject03","1-1",683,695)            #画PO的相位图