import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import math

#设置工作环境
Work_Path = "F:\\PlantarPressurePredictExperiment"
os.chdir(Work_Path)


class Distinguish:
    def __init__(self, filename, startFrame,PeakValve):
        self.filename = filename
        self.startFrame = startFrame
        self.PeakValve = PeakValve          #峰值阈值，用于截取
        self.PointArr = self.getPoint(filename, startFrame)
        self.step = 1  # 步数,舍弃第一步
        self.HeelAccelerationValve = 100        #足底加速度阈值
        self.frequency = 100                #采样率
        self.SoleAccelerationValve = 100     #脚掌加速度阈值
        self.DyanmicParaValve = 100         #动态阈值增量

        self.LeftHeelContactPointArr = np.array(self.HeelContact("Left"))
        self.RightHeelContactPointArr = np.array(self.HeelContact("Right"))

        self.LeftStepIntervalPointArr = np.array(self.GetStepInterval("Left"))
        self.RightStepIntervalPointArr = np.array(self.GetStepInterval("Right"))

        self.LeftSoleContactPointArr = np.array(self.SoleContact("Left"))
        self.RightSoleContactPointArr = np.array(self.SoleContact("Right"))

        self.LeftHeelLiftPointArr = np.array(self.HeelLift("Left"))
        self.RightHeelLiftPointArr = np.array(self.HeelLift("Right"))

        self.LeftSoleLiftPointArr = np.array(self.SoleLift("Left"))
        self.RightSoleLiftPointArr = np.array(self.SoleLift("Right"))

        self.LeftToeLiftPointArr = np.array(self.ToeLift("Left"))
        self.RightToeLiftPointArr = np.array(self.ToeLift("Right"))
        # 接下来计算
        self.LeftStepPeriodArr = np.array(self.GetStepPeriod("Left"))
        self.RightStepPeriodArr = np.array(self.GetStepPeriod("Right"))

        self.LeftHeelContactPeriodArr = np.array(self.GetHeelContactPeriod("Left"))
        self.RightHeelContactPeriodArr = np.array(self.GetHeelContactPeriod("Right"))

        self.LeftStandPhasePeriodArr = np.array(self.GetStandPhasePeriod("Left"))
        self.RightStandPhasePeriodArr = np.array(self.GetStandPhasePeriod("Right"))

        self.LeftProPhaseOfSwingPeriodArr = np.array(self.GetProPhaseOfSwingPeriod("Left"))
        self.RightProPhaseOfSwingPeriodArr = np.array(self.GetProPhaseOfSwingPeriod("Right"))

        self.LeftSwingPhasePeriodArr = np.array(self.GetSwingPhasePeriod("Left"))
        self.RightSwingPhasePeriodArr = np.array(self.GetSwingPhasePeriod("Right"))

        # self.ProPhaseOfSwingPeriodArr = np.array(self.GetProPhaseOfSwingPeriodBySole())
        # self.SwingPhasePeriodArr = np.array(self.GetSwingPhasePeriodBySole())



    def OpenFile(self, filename):
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            rows = [row for row in reader]
        Arr = np.array(rows).astype(np.float)
        return Arr

    def GetGaitStart(self, Arr, startFrame):
        return Arr[startFrame-1:, :]

    def getPoint(self, filename, startFrame):
        Arr = self.OpenFile(filename)     #获得数据列表
        Arr = self.GetGaitStart(Arr, startFrame)   #截掉起始帧前面的数据
        return np.round(Arr)             #保留两位小数

    def __LeftToRightCheck(self, Arr, index):       #判断是
        for i in range(1,6):
            if(Arr[index - i] < self.PeakValve and Arr[index + i] >self.PeakValve):
                pass
            else:
                return False
        return True

    def __FindHeelContactByAcceleration(self,Arr,index):
        try:
            CheckArr = Arr[index-50:index] - Arr[index-51:index-1]   #获得加速度列表,往前提50帧
            for i in range(CheckArr.shape[0]):
                if( CheckArr[i] > self.HeelAccelerationValve and Arr[index + i -45] > 5000):
                    return (index + i -50)
            return False
        except ValueError:
            return False

    def __FindSoleContactByAcceleration(self,Arr,start,end):
        CheckArr = Arr[start:end] - Arr[start-1:end-1]
        for i in range(len(CheckArr)):
            if(CheckArr[i] > self.SoleAccelerationValve):
                return (start+i)
        return False

    def __FindHeelLiftBySubtractNormalAverage(self,Arr,start,end):
        AverageValve = Arr[start-10:start].mean()
        #得到在这个区间的最大值，使用的方法为argpartition,可以输出对应列表a在值b的情况下的大小划分，不排序。
        #https://blog.csdn.net/qq_37007384/article/details/88668729?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-2.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-2.control
        #HeelSliceArr[np.argpartition(HeelSliceArr,len(HeelSliceArr)-1)][len(HeelSliceArr)-1]  得到最大值，类似于Sum
        HeelSliceArr = Arr[start:end]
        if(HeelSliceArr.shape[0] <= 0) : return False
        MaxIndex =   np.argpartition(HeelSliceArr,len(HeelSliceArr)-1)[len(HeelSliceArr)-1]   #得到最大值的索引

        for i in range(HeelSliceArr.shape[0] - MaxIndex):
            if(Arr[start + MaxIndex + i] <= AverageValve and Arr[start + MaxIndex + i + 1] <= AverageValve):
                return start + MaxIndex + i
        return False

    def __FindSoleLiftBySubtractNormalAverage(self,Arr,start,end):
        AverageValve = Arr[start - 10:start].mean()
        # 得到在这个区间的最大值，使用的方法为argpartition,可以输出对应列表a在值b的情况下的大小划分，不排序。
        SoleSliceArr = Arr[start:end]
        if (SoleSliceArr.shape[0] <= 0): return False
        MaxIndex = np.argpartition(SoleSliceArr, len(SoleSliceArr) - 1)[len(SoleSliceArr) - 1]  # 得到最大值的索引
        for i in range(SoleSliceArr.shape[0] - MaxIndex):
            if(Arr[start + MaxIndex + i] <= AverageValve and Arr[start + MaxIndex + i + 1] <= AverageValve) :
                return start + MaxIndex + i
        return False

    def __FindToeLiftBySubtractNormalAverage(self,Arr,start,end):
        AverageValve = Arr[start - 10:start].mean()
        # 得到在这个区间的最大值，使用的方法为argpartition,可以输出对应列表a在值b的情况下的大小划分，不排序。
        ToeSliceArr = Arr[start:end]
        if (ToeSliceArr.shape[0] <= 0): return False
        MaxIndex = np.argpartition(ToeSliceArr,len(ToeSliceArr) -1)[len(ToeSliceArr) - 1]   # 得到最大值的索引
        for i in range(ToeSliceArr.shape[0] - MaxIndex):
            if(Arr[start + MaxIndex + i] <= AverageValve + self.DyanmicParaValve  and Arr[start + MaxIndex + i + 1] <= AverageValve  + self.DyanmicParaValve):
                return start + MaxIndex + i
        return False

    def HeelContact(self,LeftOrRight):
        HeelContactArr = []
        HeelArr = self.PointArr[:, 3] if LeftOrRight == "Left" else self.PointArr[:, 10]   #确认左右脚

        for i in range(HeelArr.shape[0] - 1):
            if(HeelArr[i] > self.PeakValve):
                if(self.__LeftToRightCheck(HeelArr,i)):
                    #进来说明找到了Valve点，接下来找HeelContact点，采用往前找加速度，超过阈值的方法。
                    currentIndex = self.__FindHeelContactByAcceleration(HeelArr,i)
                    if(currentIndex != False):
                        HeelContactArr.append(currentIndex)

        return HeelContactArr

    def SoleContact(self,LeftOrRight):
        SoleContactArr = []
        #判断左右脚
        SoleArr = self.PointArr[:, 1] if LeftOrRight == "Left" else  self.PointArr[:, 8]
        StepIntervalPointArr = self.LeftStepIntervalPointArr if LeftOrRight == "Left" else self.RightStepIntervalPointArr

        for start, end in StepIntervalPointArr:
            # 对每一个脚跟着地的区间进行查找脚掌触地的点
            SoleContactIndex = self.__FindSoleContactByAcceleration(SoleArr,start,end)
            if(SoleContactIndex != False):
                SoleContactArr.append(SoleContactIndex)
        return SoleContactArr

    def HeelLift(self,LeftOrRight):
        HeelLiftArr = []
        HeelArr = self.PointArr[:, 3] if LeftOrRight == "Left" else self.PointArr[:, 10]   #确认左右脚
        StepIntervalPointArr = self.LeftStepIntervalPointArr if LeftOrRight == "Left" else self.RightStepIntervalPointArr
        for start, end in StepIntervalPointArr:
            # 对每一个脚跟着地的区间进行查找脚跟离地的区间
            HeelLiftIndex = self.__FindHeelLiftBySubtractNormalAverage(HeelArr,start,end)
            if(HeelLiftIndex != False):
                HeelLiftArr.append(HeelLiftIndex)
        return HeelLiftArr

    def SoleLift(self,LeftOrRight):
        SoleLiftArr = []
        SoleArr = self.PointArr[:, 1] if LeftOrRight == "Left" else self.PointArr[:, 8]
        StepIntervalPointArr = self.LeftStepIntervalPointArr if LeftOrRight == "Left" else self.RightStepIntervalPointArr
        for start, end in StepIntervalPointArr:
            SoleLiftIndex = self.__FindSoleLiftBySubtractNormalAverage(SoleArr, start, end)
            if (SoleLiftIndex != False):
                SoleLiftArr.append(SoleLiftIndex)
        return SoleLiftArr

    def ToeLift(self,LeftOrRight):
        ToeLiftArr = []
        ToeArr = self.PointArr[:, 0] if LeftOrRight == "Left" else self.PointArr[:, 7]
        StepIntervalPointArr = self.LeftStepIntervalPointArr if LeftOrRight == "Left" else self.RightStepIntervalPointArr
        for start, end in StepIntervalPointArr:
            ToeLiftIndex = self.__FindToeLiftBySubtractNormalAverage(ToeArr,start,end)
            if(ToeLiftIndex != False):
                ToeLiftArr.append(ToeLiftIndex)
        return ToeLiftArr

    def GetStepInterval(self,LeftOrRight):
        StepIntervalPointArr = []
        #判断左右
        HeelContactArr = self.LeftHeelContactPointArr if LeftOrRight == "Left" else self.RightHeelContactPointArr

        # 获取每一步的区间
        for i in range(len(HeelContactArr) - 1):
            StepIntervalPointArr.append([HeelContactArr[i], HeelContactArr[i + 1]])

        return StepIntervalPointArr



    def GetStepPeriod(self,LeftOrRight):
        StepPeriodArr = []
        StepIntervalPointArr = self.LeftStepIntervalPointArr if LeftOrRight == "Left" else self.RightStepIntervalPointArr
        for start, end in StepIntervalPointArr:
            StepPeriodArr.append((end - start) / self.frequency)
        return StepPeriodArr

    def GetHeelContactPeriod(self,LeftOrRight):
        SoleContactPointArr = self.LeftSoleContactPointArr if LeftOrRight == "Left" else self.RightSoleContactPointArr
        HeelContactPointArr = self.LeftHeelContactPointArr if LeftOrRight == "Left" else self.RightHeelContactPointArr

        if (len(SoleContactPointArr) != len(HeelContactPointArr)):
            print(("HeelContactPeriod LengthError:\t" + LeftOrRight + ":\t"+"HeelContactPointArr:{0}" + "\t" + "SoleContactPointArr:{1}").format(
                len(HeelContactPointArr), len(SoleContactPointArr)))
            return

        HeelContactPeriodArr = (np.array(SoleContactPointArr) - np.array(
            HeelContactPointArr)) / self.frequency
        return HeelContactPeriodArr

    def GetStandPhasePeriod(self,LeftOrRight):
        # 支撑相由脚掌着地到足跟离地的过程组成
        HeelLiftPointArr = self.LeftHeelLiftPointArr if LeftOrRight == "Left" else self.RightHeelLiftPointArr
        SoleContactPointArr = self.LeftSoleContactPointArr if LeftOrRight == "Left" else self.RightSoleContactPointArr

        if (len(SoleContactPointArr) != len(HeelLiftPointArr)):
            print(("StandPhasePeriod LengthError:\t" + LeftOrRight + ":\t"+ "SoleContactPointArr:{0}" + "\t" + "HeelLiftPointArr:{1}").format(
                len(SoleContactPointArr), len(HeelLiftPointArr)))
            return False

        StandPhasePeriodArr = (HeelLiftPointArr - SoleContactPointArr) / self.frequency
        return StandPhasePeriodArr

    def GetProPhaseOfSwingPeriod(self,LeftOrRight):
        ToeLiftPointArr = self.LeftToeLiftPointArr if LeftOrRight == "Left" else self.RightToeLiftPointArr
        HeelLiftPointArr = self.LeftHeelLiftPointArr if LeftOrRight == "Left" else self.RightHeelLiftPointArr
        if (len(ToeLiftPointArr) != len(HeelLiftPointArr)):
            print(("ProPhaseOfSwingPeriod LengthError:\t" + LeftOrRight + ":\t"+"ToeLiftPointArr:{0}" + "\t" + "HeelLiftPointArr:{1}").format(
                len(ToeLiftPointArr), len(HeelLiftPointArr)))
            return
        ProPhaseOfSwingPeriodArr = (ToeLiftPointArr - HeelLiftPointArr) / self.frequency
        return ProPhaseOfSwingPeriodArr

    def GetSwingPhasePeriod(self,LeftOrRight):
        StepIntervalPointArr = self.LeftStepIntervalPointArr if LeftOrRight == "Left" else self.RightStepIntervalPointArr
        ToeLiftPointArr = self.LeftToeLiftPointArr if LeftOrRight == "Left" else self.RightToeLiftPointArr

        if (len(StepIntervalPointArr[:, 1]) != len(ToeLiftPointArr)):
            print(("SwingPhasePeriod LengthError:\t" + LeftOrRight + ":\t"+ "StepIntervalPointArr[:,1]:{0}" + "\t" + "ToeLiftPointArr:{1}").format(
                len(StepIntervalPointArr[:, 1]), len(ToeLiftPointArr)))
            return

        SwingPhasePeriodArr = (StepIntervalPointArr[:, 1] - ToeLiftPointArr) / self.frequency
        return SwingPhasePeriodArr

    def GetProPhaseOfSwingPeriodBySole(self,LeftOrRight):
        SoleLiftPointArr = self.LeftSoleLiftPointArr if LeftOrRight == "Left" else self.RightSoleLiftPointArr
        HeelLiftPointArr = self.LeftHeelLiftPointArr if LeftOrRight == "Left" else self.RightHeelLiftPointArr

        if (len(SoleLiftPointArr) != len(HeelLiftPointArr)):
            print(("ProPhaseOfSwingPeriod LengthError:\t" + LeftOrRight + ":\t"+ "SoleLiftPointArr:{0}" + "\t" + "HeelLiftPointArr:{1}").format(
                len(SoleLiftPointArr), len(HeelLiftPointArr)))
            return

        ProPhaseOfSwingPeriodArr = (SoleLiftPointArr - HeelLiftPointArr) / self.frequency
        return ProPhaseOfSwingPeriodArr

    def GetSwingPhasePeriodBySole(self,LeftOrRight):
        StepIntervalPointArr = self.LeftStepIntervalPointArr if LeftOrRight == "Left" else self.RightStepIntervalPointArr
        SoleLiftPointArr = self.LeftSoleLiftPointArr if LeftOrRight == "Left" else self.RightSoleLiftPointArr

        if (len(StepIntervalPointArr[:, 1]) != len(SoleLiftPointArr)):
            print(("SwingPhasePeriod LengthError:\t" + LeftOrRight + ":\t"+ "StepIntervalPointArr[:,1]:{0}" + "\t" + "SoleLiftPointArr:{1}").format(
                len(StepIntervalPointArr[:, 1]), len(SoleLiftPointArr)))
            return

        SwingPhasePeriodArr = (StepIntervalPointArr[:, 1] - SoleLiftPointArr) / self.frequency
        return SwingPhasePeriodArr

    def draw(self):

        # #HeelContact
        # plt.figure(0)
        # plt.plot(self.PointArr[:, 3])
        # plt.scatter(self.LeftHeelContactPointArr, self.PointArr[:, 3][self.LeftHeelContactPointArr], marker='x', color="red")
        # plt.legend(["LHeel", "LHeelContact"])
        #
        # plt.figure(1)
        # plt.plot(self.PointArr[:, 10])
        # plt.scatter(self.RightHeelContactPointArr, self.PointArr[:, 10][self.RightHeelContactPointArr], marker='x', color="black")
        # plt.legend(["RHeel", "RHeelContact"])


        ''' #SoleContact
        plt.figure(2)
        plt.plot(self.PointArr[:,1])
        plt.plot(self.PointArr[:,3])
        plt.scatter(self.LeftSoleContactPointArr,self.PointArr[:,1][self.LeftSoleContactPointArr], marker='x', color="red")

        plt.figure(3)
        plt.plot(self.PointArr[:, 8])
        plt.plot(self.PointArr[:,10])
        plt.scatter(self.RightSoleContactPointArr,self.PointArr[:, 8][self.RightSoleContactPointArr], marker='x', color="black")
        '''

        '''  #HeelLift
        plt.figure(4)
        plt.plot(self.PointArr[:, 3])
        plt.scatter(self.LeftHeelLiftPointArr,self.PointArr[:, 3][self.LeftHeelLiftPointArr], marker='x', color="red")
        plt.legend(["LHeel", "LHeelLift"])

        plt.figure(5)
        plt.plot(self.PointArr[:, 10])
        plt.scatter(self.RightHeelLiftPointArr, self.PointArr[:, 10][self.RightHeelLiftPointArr], marker='x',color="black")
        plt.legend(["RHeel", "RHeelLift"])
        '''
        ''' #SoleLift
        plt.figure(6)
        plt.plot(self.PointArr[:, 1])
        plt.plot(self.PointArr[:, 3])
        plt.scatter(self.LeftSoleLiftPointArr, self.PointArr[:, 1][self.LeftSoleLiftPointArr], marker='x',
                    color="red")

        plt.figure(7)
        plt.plot(self.PointArr[:, 8])
        plt.plot(self.PointArr[:, 10])
        plt.scatter(self.RightSoleLiftPointArr, self.PointArr[:, 8][self.RightSoleLiftPointArr], marker='x',
                    color="black")
        '''

        # #ToeLift
        # plt.figure(8)
        # plt.plot(self.PointArr[:, 0])
        # plt.plot(self.PointArr[:, 3])
        # plt.scatter(self.LeftToeLiftPointArr, self.PointArr[:, 0][self.LeftToeLiftPointArr], marker='x',
        #             color="red")
        # plt.legend(["LToe", "LToeLift"])
        #
        # plt.figure(9)
        # plt.plot(self.PointArr[:, 7])
        # plt.plot(self.PointArr[:, 10])
        # plt.scatter(self.RightToeLiftPointArr, self.PointArr[:, 7][self.RightToeLiftPointArr], marker='x',
        #             color="black")
        # plt.legend(["RToe", "RToeLift"])


        #SumMeta1Picture
        plt.figure(10,figsize=(10,6))
        plt.plot(self.PointArr[:, 3],linestyle = ':',color = 'gray')
        #plt.plot(self.PointArr[:, 7], linestyle='--', color='yellow')
        plt.plot(self.PointArr[:, 8],linestyle = '--',color = 'blue')
        plt.plot(self.PointArr[:, 10],color = 'red')
        plt.plot(self.PointArr[:, 11],color = 'green')
        plt.scatter(self.LeftHeelContactPointArr, self.PointArr[:, 3][self.LeftHeelContactPointArr], marker='x', color="gray")
        plt.scatter(self.RightToeLiftPointArr, self.PointArr[:, 7][self.RightToeLiftPointArr], marker='x',
                                color="black")
        plt.scatter(self.RightHeelContactPointArr, self.PointArr[:, 10][self.RightHeelContactPointArr], marker='^',
                    color="red")



        plt.show()

    def writeCsv(self, name):
        # 将各个相位时间写入
        Arr = np.array(
            [self.StepPeriodArr,
             self.HeelContactPeriodArr,
             self.StandPhasePeriodArr,
             self.ProPhaseOfSwingPeriodArr,
             self.SwingPhasePeriodArr])  # 保留两位小数

        np.savetxt("test10-24\\" + name, np.round(Arr.T, 2), delimiter=",")

        return Arr.T

    def GetPoint(self):
        print("L-HeelContact")
        print(self.LeftHeelContactPointArr + 1)   #+1 是为了和Excel表对齐
        print("L-ToeLift")
        print(self.LeftToeLiftPointArr + 1)
        # print("L-SoleContact")
        # print(self.LeftSoleContactPointArr + 1)
        print("R-HeelContact")
        print(self.RightHeelContactPointArr + 1)  #+1 是为了和Excel表对齐
        # print("R-SoleContact")
        # print(self.RightSoleContactPointArr + 1)
        # print("L-HeelLift")
        # print(self.LeftHeelLiftPointArr + 1)
        # print("R-HeelLift")
        # print(self.RightHeelLiftPointArr + 1)
        # print("L-SoleLift")
        # print(self.LeftSoleLiftPointArr + 1)
        # print("R-SoleLift")
        # print(self.RightSoleLiftPointArr + 1)
        print("R-ToeLift")
        print(self.RightToeLiftPointArr + 1)

    def printPointLen(self):
        print(len(self.HeelContactPointArr))
        print(len(self.SoleContactPointArr))
        print(len(self.HeelLiftPointArr))
        print(len(self.ToeLiftPointArr))

    def printCsv(self):
        Arr = np.array(
            [self.HeelContactPeriodArr,
             self.StandPhasePeriodArr,
             self.ProPhaseOfSwingPeriodArr,
             self.SwingPhasePeriodArr])  # 保留两位小数
        return Arr.T


if __name__ == "__main__":
    filename = "SumAreaData\\subject07\\4-7.csv"
    startFrame = 1  # 根据每个样本手动选择，根据足跟选择起始帧
    PeakValve = 5000 #峰值阈值
    A = Distinguish(filename, startFrame,PeakValve)
    A.GetPoint()
    #A.draw()





















