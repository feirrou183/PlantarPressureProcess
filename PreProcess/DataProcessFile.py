import os
import numpy as np
import csv
import sys

#设置工作环境
Work_Path = "F:\\PlantarPressurePredictExperiment"
os.chdir(Work_Path)

def DataProcess(name):

    with open(name,"r") as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
    end_frame = 0
    for i in range(20,40):
        if(len(rows[i]) == 0 ) : continue
        if("END_FRAME" in rows[i][0]): end_frame = int(rows[i][0].split(" ")[1])
        if(rows[i][0] == "Frame 1"): break

    #end_frame 结束帧
    i+=1
    if(end_frame == 0): raise

    #j = range(0, 21)   #有数据的列为21列

    frames = 0
    framesList = []

    try:
        while 1:
            Array = np.array(rows[i:i+60])
            Array[Array == 'B'] = '0' #将B改为-1
            framesList.append(Array.astype(np.float))
            frames +=1
            i += 62
            if(frames >= end_frame):
                return framesList
    except IndexError:
        return framesList



def WriteFrames(FramesList,savePathAndName):
    #FramesList是一个保存了所有帧下的numpy数组的列表
    w = open(savePathAndName,'w',newline = "")
    writer = csv.writer(w)
    Array = np.array(FramesList)  #三维数组 帧数x行数x列数
    tempList = []
    #组成一个新的numpy数组,行为每帧，列为每个点
    for frame in range(Array.shape[0]):
        tempList = []      #初始化
        for row in range(Array.shape[1]):
            tempList += Array[frame][row].tolist()   #记录当前帧数
        writer.writerow(tempList) #写入当前帧
    w.close()


if __name__ == '__main__':
    for l in ['01','02','03','04','05','06','07','08','09','10','11','12']:  #1-12
        subject = "subject{}".format(l)
        for i in range(1, 12):  # 1-11
            print(subject)
            for j in range(1, 12):  # 1-11
                for k in ["L", "R"]:  # L-R
                    filenamePath = "RawData\\"+subject +"\\"+"{0}-{1}-{2}.csv".format(i, j, k)
                    writePath = "ProcessedData\\"+ subject + "\\" + "{0}-{1}-{2}.csv".format(i, j, k)
                    if (os.path.exists(filenamePath) == False): continue
                    FramesList = DataProcess(filenamePath)
                    WriteFrames(FramesList, writePath)
                    print(filenamePath)





