import os
import json
import numpy as np
import csv
import matplotlib.pyplot as plt

# 设置工作环境
Work_Path = "F:\\PlantarPressurePredictExperiment"
os.chdir(Work_Path)


Subjects = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", ]
Items = ["1", "2", "3", "4", "5"]
Sub_items = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]

#region  jsonPart
class DicManager:
    def __init__(self,DicName):
        self.DicName = DicName
        self.subjectConfigDict = self.getDic()
        self.FirstInit()

    def getDic(self):
        if (os.path.exists(self.DicName) == False):
            diction = {}
            with open(self.DicName, "w") as f:
                json.dump(diction, f)
                print("预创建成功")

        with open(self.DicName, "r") as f:
            subjectConfigDict = json.load(f)
        return  subjectConfigDict

    def FirstInit(self):
        for i in range(len(Subjects)):
            self.subjectConfigDict.setdefault("subject{}".format(Subjects[i]), {})

    def DictDump(self):
        with open(self.DicName,"w",encoding="utf-8") as f:
            json.dump(self.subjectConfigDict,f,indent=2,sort_keys=True,ensure_ascii=False)
        print("JSon处理完成")
#endregion

#region  VelocityProcessPart
class VelocityManager:
    def __init__(self,fileName):
        self.fileName = fileName
        self.LZZ,self.RZZ = self.getZZData(fileName)
        self.midVelocity = self.__TotalVelocity()


    def getZZData(self,filename):
        with open(filename,'r',encoding= "utf-8") as f:
            reader = csv.reader(f)
            count = 0
            for line in reader:
                if(len(line) >0 and line[0] == "Trajectories"):
                    break
            #找到Trajectory
            reader.__next__()   #'100'line
            itemline = reader.__next__()   #SubjectName Line
            LzzIndexList = []    #共9个index轨迹，速度，加速度
            RzzIndexList = []
            for i in range(len(itemline)):
                if(itemline[i] == 'New Subject:LZZ1'):
                    LzzIndexList.append(i)
                    LzzIndexList.append(i + 1)
                    LzzIndexList.append(i + 2)
                if(itemline[i] == 'New Subject:RZZ1'):
                    RzzIndexList.append(i)
                    RzzIndexList.append(i + 1)
                    RzzIndexList.append(i + 2)
            self.LzzIndexList = LzzIndexList
            self.RzzIndexList = RzzIndexList
            reader.__next__()   #"Frame"Line
            reader.__next__()   #"mm" Line

            LzzVelocityList = []
            RzzVelocityList = []

            for line in reader:
                if(len(line) >0):
                    if(line[LzzIndexList[0]] != ''):
                        LzzVelocityList.append(self.__calcuVelocity(line[LzzIndexList[3]],line[LzzIndexList[4]],line[LzzIndexList[5]]))
                    else:
                        LzzVelocityList.append(0)

                    if(line[RzzIndexList[0]] != ''):
                        RzzVelocityList.append(self.__calcuVelocity(line[RzzIndexList[3]],line[RzzIndexList[4]],line[RzzIndexList[5]]))
                    else:
                        RzzVelocityList.append(0)
        return LzzVelocityList,RzzVelocityList


    def __calcuVelocity(self,x,y,z):
        return  pow((pow(float(x),2)+pow(float(y),2)),0.5)



    def __TotalVelocity(self):
        if(len(self.LZZ) != len(self.RZZ)):
            print("Error",self.fileName)
            return
        midVelocity = []
        for i in range(1,len(self.LZZ)):
            if(self.LZZ[i] == 0 or self.RZZ[i] == 0):
                midVelocity.append(0)
            else:
                midVelocity.append((self.LZZ[i] + self.RZZ[i])/2)
        return midVelocity






#endregion




def ProcessVelocity(Dic):
    for i in Subjects:
        for j in Items:
            for k in Sub_items:
                subject = "subject{}".format(i)
                item = "{0}-{1}".format(j, k)
                filePath = "ViconData\\" + subject + "\\" + item + ".csv"
                if (os.path.exists(filePath)):  # 文件存在的情况下
                    velocityManager = VelocityManager(filePath)
                    Dic[subject][item] = velocityManager.midVelocity
                print("Process{}{}{}".format(i,j,k))

def Writevelocity():
    pass


if __name__ == '__main__':
    velocityDic = DicManager("SubjectViconVelocity.json")

    #plt.plot(velocityManager.midVelocity)
    #ProcessVelocity(velocityDic.subjectConfigDict)



    #velocityDic.DictDump()











