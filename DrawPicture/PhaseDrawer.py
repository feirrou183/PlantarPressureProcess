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

Subjects = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
Items = ["1", "2", "3", "4", "5"]
Sub_items = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
Strategy = ["Stright","Step","Spin"]
Step = "AP"


#region AngleDrawer
class StancePhaseDrawer:
    def __init__(self,dic,angle,Step):
        self.dic = dic
        self.angle = angle
        self.Step = Step
        self.angleItemIndexList = self.__getSubjectAngleIndex()
        self.ResultHCArr,self.ResultMSArr,self.ResultTOArr = self.DrawRectangle()


    def DrawRectangle(self):
        ResultHCArr,ResultMSArr,ResultTOArr = np.array([0 for i in range(1260)]).astype(np.float),\
                                              np.array([0 for i in range(1260)]).astype(np.float),\
                                              np.array([0 for i in range(1260)]).astype(np.float)
        for index in range(len(Subjects)):
            HCArr,MSArr,TOArr = np.array([0 for i in range(1260)]).astype(np.float),\
                                np.array([0 for i in range(1260)]).astype(np.float),\
                                np.array([0 for i in range(1260)]).astype(np.float)
            subjectName = "subject{}".format(Subjects[index])
            HCNumber = 0
            MSNumber = 0
            TONumber = 0
            for itemIndex in range(len(Sub_items)):
                itemName = "{0}-{1}".format(self.angleItemIndexList[index]+1,Sub_items[itemIndex])
                if(self.dic[subjectName]["ResultData"].__contains__(itemName)):
                    HC,SC,HL,TO = self.GetKeyPoint(subjectName,itemName,self.Step)
                    TempHCArr,TempMSArr,TempTOArr = self.GetArr(subjectName,itemName,self.Step,HC,SC,HL,TO)
                    if(len(TempHCArr) == 1260 and TempHCArr[0] != np.nan):
                        HCArr += TempHCArr
                        HCNumber +=1
                    if(len(TempMSArr) == 1260 and TempMSArr[0] != np.nan):
                        MSArr += TempMSArr
                        MSNumber +=1
                    if (len(TempTOArr) == 1260 and TempTOArr[0] != np.nan):
                        TOArr += TempTOArr
                        TONumber +=1

                    print(subjectName,"--",itemName)

            HCArr /= HCNumber
            MSArr /= MSNumber
            TOArr /= TONumber
            print("Number --->" , HCNumber,"--",MSNumber,"--",TONumber)

            ResultHCArr += HCArr
            ResultMSArr += MSArr
            ResultTOArr += TOArr


        ResultHCArr /= 12
        ResultMSArr /= 12
        ResultTOArr /= 12

        return ResultHCArr,ResultMSArr,ResultTOArr

    def __getSubjectAngleIndex(self):
        indexList = []
        for sub in Subjects:
            angleStr = self.dic["subject{}".format(sub)]["ExperimentTurn"]
            angleOrder = list(map(self.__getAngleOrder,angleStr))
            angleOrder.remove(None)
            angleIndex = angleOrder.index(self.angle)
            indexList.append(angleIndex)
        return  indexList

    def __getAngle(self,subjectsName):
        return

    def __getAngleOrder(self,number):
        if (number == '3'):
            return 30
        if (number == '6'):
            return 60
        if (number == '9'):
            return 90
        if (number == '1'):
            return 120
        if (number == '0'):
            return 0


    def GetKeyPoint(self,subject,item,step):
        HC = self.dic[subject]["ResultData"][item][step].split(",")[0]
        SC = self.dic[subject]["ResultData"][item]["{0}SCValue".format(step)].split("_")[0]
        HL = self.dic[subject]["ResultData"][item]["{0}HLValue".format(step)].split("_")[0]
        TO = self.dic[subject]["ResultData"][item][step].split(",")[1]
        return int(HC),int(SC),int(HL),int(TO)

    def GetArr(self,subject,detailItemName,eachStep,HC,SC,HL,TO):
        file = "1"
        if(eachStep == "BTS" or eachStep == "TS"):
            file = open("ProcessedData\\{0}\\{1}-L.csv".format(subject,detailItemName),encoding="utf-8")
        elif(eachStep == "BAP" or eachStep == "AP" or eachStep == "DS"):
            file = open("ProcessedData\\{0}\\{1}-R.csv".format(subject,detailItemName),encoding="utf-8")
        else:
            print("Error",subject,detailItemName)

        Arr = np.loadtxt(file,delimiter = ",")
        HCArr = Arr[HC-1:SC-1,:]
        MSArr = Arr[SC:HL-1,:]
        TOArr = Arr[HL:TO-1,:]
        file.close()

        if(len(HCArr) >1): HCArr = np.average(HCArr, axis= 0)
        if(len(MSArr) >1): MSArr = np.average(MSArr, axis= 0)
        if(len(TOArr) >1): TOArr = np.average(TOArr, axis= 0)

        return HCArr,MSArr,TOArr


    def Draw(self,pltstyle):
        plt.style.use(pltstyle)
        cmap = cm.get_cmap('jet')
        plt.rcParams["axes.grid"] = False
        kernel = np.ones((2,2), np.float32) / 4

        Hc  = self.ResultHCArr.reshape(60,21)
        Ms  = self.ResultMSArr.reshape(60,21)
        To  = self.ResultTOArr.reshape(60,21)

        Hc = cv.filter2D(Hc,-1,kernel)
        Ms = cv.filter2D(Ms,-1,kernel)
        To = cv.filter2D(To,-1,kernel)

        plt.figure(0,[4.2,12])
        plt.imshow(Hc,cmap=cmap)

        plt.title("ResultHCArr")

        plt.figure(1,[4.2,12])
        plt.imshow(Ms,cmap=cmap)
        plt.title("ResultMSArr")

        plt.figure(2,[4.2,12])
        plt.imshow(To,cmap=cmap)
        plt.title("ResultTOArr")
#endregion

#region  StrategyDrawer
class StancePhaseStrategyDrawer:
    def __init__(self,dic,strategy,Step):
        self.dic = dic
        self.strategy = strategy
        self.Step = Step
        self.ResultHCArr,self.ResultMSArr,self.ResultTOArr = self.DrawRectangle()


    def DrawRectangle(self):
        ResultHCArr,ResultMSArr,ResultTOArr = np.array([0 for i in range(1260)]).astype(np.float),\
                                              np.array([0 for i in range(1260)]).astype(np.float),\
                                              np.array([0 for i in range(1260)]).astype(np.float)
        for index in range(len(Subjects)):
            HCArr,MSArr,TOArr = np.array([0 for i in range(1260)]).astype(np.float),\
                                np.array([0 for i in range(1260)]).astype(np.float),\
                                np.array([0 for i in range(1260)]).astype(np.float)
            subjectName = "subject{}".format(Subjects[index])
            HCNumber = 0
            MSNumber = 0
            TONumber = 0
            for item in Items:
                for itemIndex in range(len(Sub_items)):
                    itemName = "{0}-{1}".format(item,itemIndex)
                    if(self.dic[subjectName]["ResultData"].__contains__(itemName) and
                       self.dic[subjectName]["ResultData"][itemName]["strategy"] == self.strategy and
                       self.dic[subjectName]["ResultData"][itemName]["angle"] != "0°"
                    ):
                        HC,SC,HL,TO = self.GetKeyPoint(subjectName,itemName,self.Step)
                        TempHCArr,TempMSArr,TempTOArr = self.GetArr(subjectName,itemName,self.Step,HC,SC,HL,TO)
                        if(len(TempHCArr) == 1260 and TempHCArr[0] != np.nan):
                            HCArr += TempHCArr
                            HCNumber +=1
                        if(len(TempMSArr) == 1260 and TempMSArr[0] != np.nan):
                            MSArr += TempMSArr
                            MSNumber +=1
                        if (len(TempTOArr) == 1260 and TempTOArr[0] != np.nan):
                            TOArr += TempTOArr
                            TONumber +=1

                        print(subjectName,"--",itemName)

            if(HCNumber > 0) :
                HCArr /= HCNumber
                ResultHCArr += HCArr
            if(MSNumber > 0) :
                MSArr /= MSNumber
                ResultMSArr += MSArr
            if(TONumber > 0) :
                TOArr /= TONumber
                ResultTOArr += TOArr
            print("Number --->" , HCNumber,"--",MSNumber,"--",TONumber)





        ResultHCArr /= 12
        ResultMSArr /= 12
        ResultTOArr /= 12

        return ResultHCArr,ResultMSArr,ResultTOArr

    def GetKeyPoint(self,subject,item,step):
        HC = self.dic[subject]["ResultData"][item][step].split(",")[0]
        SC = self.dic[subject]["ResultData"][item]["{0}SCValue".format(step)].split("_")[0]
        HL = self.dic[subject]["ResultData"][item]["{0}HLValue".format(step)].split("_")[0]
        TO = self.dic[subject]["ResultData"][item][step].split(",")[1]
        return int(HC),int(SC),int(HL),int(TO)

    def GetArr(self,subject,detailItemName,eachStep,HC,SC,HL,TO):
        file = "1"
        if(eachStep == "BTS" or eachStep == "TS"):
            file = open("ProcessedData\\{0}\\{1}-L.csv".format(subject,detailItemName),encoding="utf-8")
        elif(eachStep == "BAP" or eachStep == "AP" or eachStep == "DS"):
            file = open("ProcessedData\\{0}\\{1}-R.csv".format(subject,detailItemName),encoding="utf-8")
        else:
            print("Error",subject,detailItemName)

        Arr = np.loadtxt(file,delimiter = ",")
        HCArr = Arr[HC-1:SC-1,:]
        MSArr = Arr[SC:HL-1,:]
        TOArr = Arr[HL:TO-1,:]
        file.close()

        if(len(HCArr) >1): HCArr = np.average(HCArr, axis= 0)
        if(len(MSArr) >1): MSArr = np.average(MSArr, axis= 0)
        if(len(TOArr) >1): TOArr = np.average(TOArr, axis= 0)

        return HCArr,MSArr,TOArr

    def Draw(self,pltstyle):
        plt.style.use(pltstyle)
        cmap = cm.get_cmap('jet')
        plt.rcParams["axes.grid"] = False
        kernel = np.ones((2,2), np.float32) / 4

        Hc  = self.ResultHCArr.reshape(60,21)
        Ms  = self.ResultMSArr.reshape(60,21)
        To  = self.ResultTOArr.reshape(60,21)

        Hc = cv.filter2D(Hc,-1,kernel)
        Ms = cv.filter2D(Ms,-1,kernel)
        To = cv.filter2D(To,-1,kernel)

        plt.figure(0,[4.2,12])
        plt.imshow(Hc,cmap=cmap)

        plt.title("ResultHCArr")

        plt.figure(1,[4.2,12])
        plt.imshow(Ms,cmap=cmap)
        plt.title("ResultMSArr")

        plt.figure(2,[4.2,12])
        plt.imshow(To,cmap=cmap)
        plt.title("ResultTOArr")

#endregion



if __name__ == '__main__':
    Dic = dicManager.DicManager("subjectConfig.json").subjectConfigDict
    #stancePhaseDrawer = StancePhaseDrawer(Dic,90,"AP")
    stancePhaseDrawer =  StancePhaseStrategyDrawer(Dic,"Spin","TS")
    stancePhaseDrawer.Draw("ggplot")











