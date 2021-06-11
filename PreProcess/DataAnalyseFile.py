import csv
import numpy as np
#import matplotlib.pyplot as plt
import json
import os

#设置工作环境
Work_Path = "/"
os.chdir(Work_Path)





class GetAnalyse:
    def __init__(self,Subject,filePath):
        #filePath 类型：“xxx\\xxx\\1-1-”
        self.filePath = filePath
        self.subject = Subject
        self.GetPlantarArea()

        #获取左脚累加和
        self.Arr = self.openFile("L")
        self.LToeArr = self.GetToeSum(self.LeftToeArea)
        self.LMetaArr = self.GetMetaSum(self.LeftToeArea,self.LeftMetaArea)
        self.LArchArr = self.GetArchSum(self.LeftMetaArea,self.LeftArchArea)
        self.LHeelArr = self.GetHeelSum(self.LeftArchArea,self.LeftHeelArea)
        self.LMeta1,self.LMeta2,self.LMeta3 = self.GetMetaDerived(self.LeftToeArea,self.LeftMetaArea)

        #获取右脚累加和
        self.Arr = self.openFile("R")
        self.RToeArr = self.GetToeSum(self.RightToeArea)
        self.RMetaArr = self.GetMetaSum(self.RightToeArea, self.RightMetaArea)
        self.RArchArr = self.GetArchSum(self.RightMetaArea, self.RightArchArea)
        self.RHeelArr = self.GetHeelSum(self.RightArchArea, self.RightHeelArea)
        self.RMeta1, self.RMeta2, self.RMeta3 = self.GetMetaDerived(self.RightToeArea, self.RightMetaArea)



    def openFile(self,LeftOrRight):
        with open(self.filePath + LeftOrRight +".csv", 'r') as f:
            reader = csv.reader(f)
            rows = [row for row in reader]
        Arr = np.array(rows).astype(np.float)
        return Arr


    def GetPlantarArea(self):
        with open("subjectConfig.json", "r") as f:
            subjectConfigDict = json.load(f)
            self.LeftToeArea = subjectConfigDict[self.subject]["PlantarAreaNums"]["LeftToeArea"]
            self.LeftMetaArea = subjectConfigDict[self.subject]["PlantarAreaNums"]["LeftMetaArea"]
            self.LeftArchArea = subjectConfigDict[self.subject]["PlantarAreaNums"]["LeftArchArea"]
            self.LeftHeelArea = subjectConfigDict[self.subject]["PlantarAreaNums"]["LeftHeelArea"]

            self.RightToeArea = subjectConfigDict[self.subject]["PlantarAreaNums"]["RightToeArea"]
            self.RightMetaArea = subjectConfigDict[self.subject]["PlantarAreaNums"]["RightMetaArea"]
            self.RightArchArea = subjectConfigDict[self.subject]["PlantarAreaNums"]["RightArchArea"]
            self.RightHeelArea = subjectConfigDict[self.subject]["PlantarAreaNums"]["RightHeelArea"]

    def GetToeSum(self,ToeArea):
        arr = self.Arr[:,0:ToeArea]
        Toe = np.sum(arr,1)                #按行加
        return Toe

    def GetMetaSum(self,ToeArea,MetaArea):
        arr = self.Arr[:,ToeArea:MetaArea]
        Meta = np.sum(arr,1)                #按行加
        return Meta

    def GetArchSum(self,MetaArea,ArchArea):
        arr = self.Arr[:, MetaArea:ArchArea]
        Arch = np.sum(arr, 1)  # 按行加
        return Arch

    def GetHeelSum(self,ArchArea,HeelArea):
        arr = self.Arr[:, ArchArea:HeelArea]
        Heel = np.sum(arr, 1)  # 按行加
        return Heel

    def GetMetaDerived(self,ToeArea,MetaArea):
        arr = self.Arr[:, ToeArea:MetaArea]
        lines = (MetaArea-ToeArea)//21
        Meta1 =np.zeros(arr.shape[0])
        Meta2 = np.zeros(arr.shape[0])
        Meta3 = np.zeros(arr.shape[0])
        for i in range(lines):
            m = i*21          #行数
            Meta1 += np.sum(arr[:, m:m+7], 1)
            Meta2 += np.sum(arr[:, m+7:m+14], 1)
            Meta3 += np.sum(arr[:, m+14:m+21], 1)

        return Meta1,Meta2,Meta3

    def WriteSumData(self,filePath):
        Arr = np.vstack((self.LToeArr,self.LMetaArr,self.LArchArr,self.LHeelArr,
                         self.LMeta1,self.LMeta2,self.LMeta3,
                         self.RToeArr,self.RMetaArr,self.RArchArr,self.RHeelArr,
                         self.RMeta1,self.RMeta2,self.RMeta3
                         ))
        Arr = np.around(Arr.T,2)

        np.savetxt(filePath,Arr ,delimiter=",", fmt='%.2f')  #保留2位小数
        print(self.subject)




if __name__ == '__main__':
    for i in ['10','11','12']:  #1-9
        print(i)
        for j in range(1,12):  #1-11
            for k in range(1,12): #1-11
                subject = "subject{}".format(i)
                filePath ="ProcessedData\\" + subject + "\\{0}-{1}-".format(j,k)
                writePath = "SumAreaData\\" + subject + "\\{0}-{1}.csv".format(j,k)
                if(os.path.exists(filePath + "L.csv")):  #文件存在的情况下
                    DataAnalyse = GetAnalyse("subject01",filePath)
                    DataAnalyse.WriteSumData(writePath)
                    print(subject+"-"+"{0}-{1}".format(j,k))


    '''
    def draw(self):
        plt.figure()
        plt.plot(self.ToeArr,color = "red")

        plt.plot(self.MetaArr,color = "blue")

        plt.plot(self.ArchArr,color = "gray")

        plt.plot(self.HeelArr,color = "green")
        plt.legend(["Toe" ,"Meta","Arch","Heel"])

    def drawMeta(self):
        plt.figure()
        plt.plot(self.Meta1, color="red")

        plt.plot(self.Meta2, color="blue")

        plt.plot(self.Meta3, color="gray")

        plt.plot(self.MetaArr,color = "green")

        plt.legend(["Meta1", "Meta2", "Meta3", "Meta"])

    '''












