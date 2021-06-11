import os
import json
import numpy as np
import csv
import matplotlib.pyplot as plt
import ProcessProgram.SelfTool.SlidingAverageFilter as slf
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
        self.zzPositionList = self.getZZData(fileName)
        self.Velocity = self.__TotalVelocity()


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

            ZZ_position_list = []

            for line in reader:
                if(len(line) >0):
                    if(line[LzzIndexList[0]] != '' and line[RzzIndexList[0]] != ''):
                        ZZ_position_list.append(self.__calcuPosition(line[LzzIndexList[0]],line[LzzIndexList[1]],
                                                                     line[RzzIndexList[0]],line[RzzIndexList[1]]))
                    else:
                        if(len(ZZ_position_list) >1):
                            ZZ_position_list.append([ZZ_position_list[-1][0],ZZ_position_list[-1][1]])
                        else:
                            print("Error",self.fileName)
                            return [[0,0],[0,0]]

        return ZZ_position_list


    def __calcuPosition(self,Lzzx,Lzzy,Rzzx,Rzzy):
        x = (float(Lzzx) +float(Rzzx)) / 2
        y = (float(Lzzy) +float(Rzzy)) / 2
        return [x,y]




    def __TotalVelocity(self):
        allFrame = len(self.zzPositionList)
        TrajectoryLen = 0
        x0,y0 = self.zzPositionList[0][0],self.zzPositionList[0][1]
        for x,y in self.zzPositionList:
            sliceLen = pow(pow(x - x0, 2) + pow(y - y0, 2), 0.5)
            x0 = x
            y0 = y
            TrajectoryLen += sliceLen

        Velocity = TrajectoryLen / allFrame * 100
        return Velocity

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
                    Dic[subject][item] = velocityManager.Velocity
                #print("Process{}{}{}".format(i,j,k))

def Writevelocity():
    pass

#region ExportToExcel

def WriteCsv(dic,fileName):
    with open(fileName, 'w',newline='') as f:
        writer = csv.writer(f)
        for i in Subjects:
            for j in Items:
                for k in Sub_items:
                    subject = "subject{}".format(i)
                    item = "{0}-{1}".format(j, k)
                    line = []
                    line.append(subject)
                    line.append(item)
                    if(dic[subject].__contains__(item)):
                        line.append(dic[subject][item])
                        writer.writerow(line)
                    print(subject,"---",item)
#endregion

if __name__ == '__main__':
    velocityDic = DicManager("SubjectViconVelocity.json")

    #plt.plot(velocityManager.midVelocity)
    #ProcessVelocity(velocityDic.subjectConfigDict)

    WriteCsv(velocityDic.subjectConfigDict,"OutPut\\TrajectoryVelocity.csv")



    #velocityDic.DictDump()











