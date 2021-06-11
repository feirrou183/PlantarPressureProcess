import os

# 设置工作环境
Work_Path = "F:\\PlantarPressurePredictExperiment"
os.chdir(Work_Path)
import ProcessProgram.FITDataAnalyse as FIT
import csv
import numpy as np
from enum import Enum

import ProcessProgram.SelfTool.FilterTool as Filter

# 需要信息
'''
    X: Heel,Arch,Sole,Toe,Meta1,Meta2,Meta3   Y: BAP,AP,TS,DS    Z：Global,HC,MS,TF
1. 全局各个区块的X_Y_ZResultData
2. HC各个区块的X_Y_ZResultData     
3. MS各个区块的X_Y_ZResultData
4. TF各个区块的X_Y_ZResultData
'''
class StepType(Enum):
    BAP = 0
    BTS = 1
    AP = 2
    TS = 3
    DS = 4

class ProcessData():
    def __init__(self, filepath, BAP, BTS, AP, TS, DS, Weight):
        try:
            self.filepath = filepath
            with open(filepath, 'r') as f:
                reader = csv.reader(f)
                rows = [row for row in reader]
                self.allData = np.array(rows).astype(float)
            self.Weight = Weight

            self.BAPArr = self.allData[BAP[0] - 1:BAP[1] - 2, 7:]
            self.BTSArr = self.allData[BTS[0] - 1:BTS[1] - 2, :7]
            self.APArr = self.allData[AP[0] - 1:AP[1] - 2, 7:]
            self.TSArr = self.allData[TS[0] - 1:TS[1] - 2, :7]
            self.DSArr = self.allData[DS[0] - 1:DS[1] - 2, 7:]

            self.SoleAccelerationValve = 20  # 脚掌加速度阈值
            self.frequency = 100
            self.HeelAccelerationValve = 100  # 足底加速度阈值
            self.DyanmicParaValve = 100  # 动态阈值增量
            self.SoleRiseValve = 1000
            self.filterNumber = 5 #滑动平均窗数量
            self.ErrorFlag = False
            self.BAPErrorFlag = False
            self.BTSErrorFlag = False
            self.APErrorFlag = False
            self.TSErrorFlag = False
            self.DSErrorFlag = False

            self.multiple = 1.5


            self.BAPHCList, self.BAPMSList, self.BAPTFList,self.BAPSCValue,self.BAPHLValue = self.GetPhaseList(BAP, self.allData[:, 7:],StepType.BAP)
            self.BAP_HCPeriod, self.BAP_MSPeriod, self.BAP_TFPeriod = self.GetPeriod(self.BAPHCList, self.BAPMSList,self.BAPTFList)
            self.BAPPeriod = self.BAP_HCPeriod + self.BAP_MSPeriod + self.BAP_TFPeriod

            self.BTSHCList, self.BTSMSList, self.BTSTFList,self.BTSSCValue, self.BTSHLValue = self.GetPhaseList(BTS, self.allData[:, :7],StepType.BTS)
            self.BTS_HCPeriod, self.BTS_MSPeriod, self.BTS_TFPeriod = self.GetPeriod(self.BTSHCList, self.BTSMSList,self.BTSTFList)
            self.BTSPeriod = self.BTS_HCPeriod + self.BTS_MSPeriod + self.BTS_TFPeriod

            self.APHCList, self.APMSList, self.APTFList,self.APSCValue, self.APHLValue = self.GetPhaseList(AP,self.allData[:, 7:],StepType.AP)
            self.AP_HCPeriod, self.AP_MSPeriod, self.AP_TFPeriod = self.GetPeriod(self.APHCList, self.APMSList,self.APTFList)
            self.APPeriod = self.BTS_HCPeriod + self.BTS_MSPeriod + self.BTS_TFPeriod

            self.TSHCList, self.TSMSList, self.TSTFList,self.TSSCValue, self.TSHLValue = self.GetPhaseList(TS,self.allData[:, :7],StepType.TS)
            self.TS_HCPeriod, self.TS_MSPeriod, self.TS_TFPeriod = self.GetPeriod(self.TSHCList, self.TSMSList,self.TSTFList)
            self.TSPeriod = self.TS_HCPeriod + self.TS_MSPeriod + self.TS_TFPeriod

            self.DSHCList, self.DSMSList, self.DSTFList,self.DSSCValue, self.DSHLValue = self.GetPhaseList(DS,self.allData[:, 7:],StepType.DS)
            self.DS_HCPeriod, self.DS_MSPeriod, self.DS_TFPeriod = self.GetPeriod(self.DSHCList, self.DSMSList,self.DSTFList)
            self.DSPeriod = self.DS_HCPeriod+ self.DS_MSPeriod+ self.DS_TFPeriod

            # 全局 全脚 ResultData
            self.Total_BAP_GlobalResultData, self.Total_BTS_GlobalResultData, self.Total_AP_GlobalResultData, self.Total_TS_GlobalResultData, \
            self.Total_DS_GlobalResultData = self.GetTotalGlobalResultData()

            # Global ResultData
            self.Heel_BAP_GlobalResultData, self.Arch_BAP_GlobalResultData, self.Sole_BAP_GlobalResultData, \
            self.Toe_BAP_GlobalResultData, self.Meta1_BAP_GlobalResultData, self.Meta2_BAP_GlobalResultData, \
            self.Meta3_BAP_GlobalResultData = self.GetGlobalResultData(self.BAPArr)

            self.Heel_BTS_GlobalResultData, self.Arch_BTS_GlobalResultData, self.Sole_BTS_GlobalResultData, \
            self.Toe_BTS_GlobalResultData, self.Meta1_BTS_GlobalResultData, self.Meta2_BTS_GlobalResultData, \
            self.Meta3_BTS_GlobalResultData = self.GetGlobalResultData(self.BTSArr)

            self.Heel_AP_GlobalResultData, self.Arch_AP_GlobalResultData, self.Sole_AP_GlobalResultData, \
            self.Toe_AP_GlobalResultData, self.Meta1_AP_GlobalResultData, self.Meta2_AP_GlobalResultData, \
            self.Meta3_AP_GlobalResultData = self.GetGlobalResultData(self.APArr)

            self.Heel_TS_GlobalResultData, self.Arch_TS_GlobalResultData, self.Sole_TS_GlobalResultData, \
            self.Toe_TS_GlobalResultData, self.Meta1_TS_GlobalResultData, self.Meta2_TS_GlobalResultData, \
            self.Meta3_TS_GlobalResultData = self.GetGlobalResultData(self.TSArr)

            self.Heel_DS_GlobalResultData, self.Arch_DS_GlobalResultData, self.Sole_DS_GlobalResultData, \
            self.Toe_DS_GlobalResultData, self.Meta1_DS_GlobalResultData, self.Meta2_DS_GlobalResultData, \
            self.Meta3_DS_GlobalResultData = self.GetGlobalResultData(self.DSArr)

            # HC ResultData
            self.Heel_BAP_HCResultData, self.Arch_BAP_HCResultData, self.Sole_BAP_HCResultData, \
            self.Toe_BAP_HCResultData, self.Meta1_BAP_HCResultData, self.Meta2_BAP_HCResultData, \
            self.Meta3_BAP_HCResultData = self.GetPhaseResultData(self.BAPHCList, self.BAPArr)

            self.Heel_BTS_HCResultData, self.Arch_BTS_HCResultData, self.Sole_BTS_HCResultData, \
            self.Toe_BTS_HCResultData, self.Meta1_BTS_HCResultData, self.Meta2_BTS_HCResultData, \
            self.Meta3_BTS_HCResultData = self.GetPhaseResultData(self.BTSHCList, self.BTSArr)

            self.Heel_AP_HCResultData, self.Arch_AP_HCResultData, self.Sole_AP_HCResultData, \
            self.Toe_AP_HCResultData, self.Meta1_AP_HCResultData, self.Meta2_AP_HCResultData, \
            self.Meta3_AP_HCResultData = self.GetPhaseResultData(self.APHCList, self.APArr)

            self.Heel_TS_HCResultData, self.Arch_TS_HCResultData, self.Sole_TS_HCResultData, \
            self.Toe_TS_HCResultData, self.Meta1_TS_HCResultData, self.Meta2_TS_HCResultData, \
            self.Meta3_TS_HCResultData = self.GetPhaseResultData(self.TSHCList, self.TSArr)

            self.Heel_DS_HCResultData, self.Arch_DS_HCResultData, self.Sole_DS_HCResultData, \
            self.Toe_DS_HCResultData, self.Meta1_DS_HCResultData, self.Meta2_DS_HCResultData, \
            self.Meta3_DS_HCResultData = self.GetPhaseResultData(self.DSHCList, self.DSArr)

            # MS ResultData
            self.Heel_BAP_MSResultData, self.Arch_BAP_MSResultData, self.Sole_BAP_MSResultData, \
            self.Toe_BAP_MSResultData, self.Meta1_BAP_MSResultData, self.Meta2_BAP_MSResultData, \
            self.Meta3_BAP_MSResultData = self.GetPhaseResultData(self.BAPMSList, self.BAPArr)

            self.Heel_BTS_MSResultData, self.Arch_BTS_MSResultData, self.Sole_BTS_MSResultData, \
            self.Toe_BTS_MSResultData, self.Meta1_BTS_MSResultData, self.Meta2_BTS_MSResultData, \
            self.Meta3_BTS_MSResultData = self.GetPhaseResultData(self.BTSMSList, self.BTSArr)

            self.Heel_AP_MSResultData, self.Arch_AP_MSResultData, self.Sole_AP_MSResultData, \
            self.Toe_AP_MSResultData, self.Meta1_AP_MSResultData, self.Meta2_AP_MSResultData, \
            self.Meta3_AP_MSResultData = self.GetPhaseResultData(self.APMSList, self.APArr)

            self.Heel_TS_MSResultData, self.Arch_TS_MSResultData, self.Sole_TS_MSResultData, \
            self.Toe_TS_MSResultData, self.Meta1_TS_MSResultData, self.Meta2_TS_MSResultData, \
            self.Meta3_TS_MSResultData = self.GetPhaseResultData(self.TSMSList, self.TSArr)

            self.Heel_DS_MSResultData, self.Arch_DS_MSResultData, self.Sole_DS_MSResultData, \
            self.Toe_DS_MSResultData, self.Meta1_DS_MSResultData, self.Meta2_DS_MSResultData, \
            self.Meta3_DS_MSResultData = self.GetPhaseResultData(self.DSMSList, self.DSArr)

            # TF ResultData
            self.Heel_BAP_TFResultData, self.Arch_BAP_TFResultData, self.Sole_BAP_TFResultData, \
            self.Toe_BAP_TFResultData, self.Meta1_BAP_TFResultData, self.Meta2_BAP_TFResultData, \
            self.Meta3_BAP_TFResultData = self.GetPhaseResultData(self.BAPTFList, self.BAPArr)

            self.Heel_BTS_TFResultData, self.Arch_BTS_TFResultData, self.Sole_BTS_TFResultData, \
            self.Toe_BTS_TFResultData, self.Meta1_BTS_TFResultData, self.Meta2_BTS_TFResultData, \
            self.Meta3_BTS_TFResultData = self.GetPhaseResultData(self.BTSTFList, self.BTSArr)

            self.Heel_AP_TFResultData, self.Arch_AP_TFResultData, self.Sole_AP_TFResultData, \
            self.Toe_AP_TFResultData, self.Meta1_AP_TFResultData, self.Meta2_AP_TFResultData, \
            self.Meta3_AP_TFResultData = self.GetPhaseResultData(self.APTFList, self.APArr)

            self.Heel_TS_TFResultData, self.Arch_TS_TFResultData, self.Sole_TS_TFResultData, \
            self.Toe_TS_TFResultData, self.Meta1_TS_TFResultData, self.Meta2_TS_TFResultData, \
            self.Meta3_TS_TFResultData = self.GetPhaseResultData(self.TSTFList, self.TSArr)

            self.Heel_DS_TFResultData, self.Arch_DS_TFResultData, self.Sole_DS_TFResultData, \
            self.Toe_DS_TFResultData, self.Meta1_DS_TFResultData, self.Meta2_DS_TFResultData, \
            self.Meta3_DS_TFResultData = self.GetPhaseResultData(self.DSTFList, self.DSArr)

        except IndexError:
            print(self.filepath)
            print(len(self.BAPArr),BAP[0])
            print(len(self.BTSArr),BTS[0])
            print(len(self.APArr),AP[0])
            print(len(self.TSArr), TS[0])
            print(len(self.DSArr),DS[0])
            raise IndexError

    def __contains__(self, item):
        try:
            eval(" a = self.{}".format(item))
            return True
        except AttributeError:
            return False

    def GetSCHLValue(self,HCList,MSList):
        return int(HCList[1]),int(MSList[1])
    def ErrorToChangeFlag(self,stepType):
        if (stepType == StepType.BAP):
            self.BAPErrorFlag = True

        if (stepType == StepType.BTS):
            self.BTSErrorFlag = True

        if (stepType == StepType.AP):
            self.APErrorFlag = True

        if (stepType == StepType.TS):
            self.TSErrorFlag = True

        if (stepType == StepType.DS):
            self.DSErrorFlag = True

        if(self.ErrorFlag == False):
            self.ErrorFlag = True
            return True
        return False


    def GetPhaseList(self, IndexList, AllData,stepType):

        '''
        :param IndexList: 对应与AllData的IndexList
        :param AllData:  n行7列 计算单脚全数据
        :return: [HCStart,SCStart],[SCStart,HeelOFF],[HeelOFF,ToeOFF]:
        '''

        ToeList = AllData[:, 0]
        SoleList = AllData[:, 1]
        ArchList = AllData[:, 2]
        HeelList = AllData[:, 3]
        StartIndex = IndexList[0]
        EndIndex = IndexList[1] - 1

        SoleContactIndex = self.__FindSoleContactByAcceleration(StartIndex, EndIndex, SoleList,stepType)
        #HeelLiftIndex = self.__FindHeelLiftBySubtractNormalAverage(StartIndex, EndIndex, HeelList,stepType)
        HeelLiftIndex = self.__FindHeelLiftBySoleList(StartIndex, EndIndex, SoleList,HeelList,ToeList,stepType)

        SCValue = str(StartIndex + SoleContactIndex)+ str("_") + str(SoleList[StartIndex+SoleContactIndex]) + str("_") + str(int(SoleList[StartIndex+SoleContactIndex + 1] - SoleList[StartIndex+SoleContactIndex]))
        HLValue = str(StartIndex + HeelLiftIndex)+ str("_") + str(HeelList[StartIndex + HeelLiftIndex]) + str("_") + str(int(HeelList[StartIndex + HeelLiftIndex + 1] - SoleList[StartIndex+SoleContactIndex]))
        return [0,SoleContactIndex],[SoleContactIndex,HeelLiftIndex],[HeelLiftIndex,EndIndex - StartIndex - 1],\
               SCValue,HLValue




    def __FindHeelLiftBySoleList(self, StartIndex, EndIndex,SoleArr,HeelArr,TorArr,stepType):

        # 思路:
        # 1.  从脚掌峰值往前推15帧开始，到SoleMax点截止
        # 2.  速度变化率第一个小于的点开始计算
        # 3.  足跟开始点往后推4帧取平均值HeelValueValve。 当前值小于平均值V_check则认为该点为足跟离地点

        SoleProcessArr = SoleArr[StartIndex - 20: EndIndex]
        SoleProcessFilterArr = np.array(Filter.SlideAvgfliter(SoleProcessArr, self.filterNumber))
        SoleFiltedArr = SoleProcessFilterArr[19:]
        SoleList = SoleFiltedArr[1:]
        SoleMaxIndex = np.argpartition(SoleList, len(SoleList) - 1)[len(SoleList) - 1]  # 得到最大值的索引

        HeelProcessArr = HeelArr[StartIndex - 20: EndIndex]
        HeelProcessFilterArr = np.array(Filter.SlideAvgfliter(HeelProcessArr, self.filterNumber))
        HeelFiltedArr = HeelProcessFilterArr[19:]
        AccelerationList = HeelFiltedArr[1:] - HeelFiltedArr[0:-1]  # 得到滤波后的加速度列表
        HeelList = HeelFiltedArr[1:]
        #HeelMaxIndex = np.argpartition(HeelList, len(HeelList) - 1)[len(HeelList) - 1]  # 得到最大值的索引

        HeelValve = np.average(np.abs(HeelList[0:4]))
        startCheckIndex = SoleMaxIndex - 15

        for i in range(startCheckIndex,SoleMaxIndex):
            if(abs(AccelerationList[i]) <= 150 and HeelList[i] < HeelValve):
                return i

        if (self.ErrorToChangeFlag(stepType)):
            print("HeelLift Error!!\t \t \t!!!!!!!!!!!!!!!!!--->", self.filepath, "--StartIndex：", StartIndex)
        return SoleMaxIndex - 5


    def __FindHeelLiftBySubtractNormalAverage(self, StartIndex, EndIndex, Arr,stepType):

        ProcessArr = Arr[StartIndex - 20: EndIndex]
        ProcessFilterArr = np.array(Filter.SlideAvgfliter(ProcessArr, self.filterNumber))
        FiltedArr = ProcessFilterArr[19:]
        AccelerationList = FiltedArr[1:] - FiltedArr[0:-1]  # 得到滤波后的加速度列表
        HeelList = FiltedArr[1:]

        AccelerationValve = np.average(AccelerationList[-10:])

        HeelLiftValve = max(np.average(ProcessFilterArr[:10]), np.max(ProcessFilterArr[-10:]))  # 取脚跟前10与后10中较大的值作为均值阈值

        DyanmicParaValve = HeelLiftValve * self.multiple
        MaxIndex = np.argpartition(HeelList, len(HeelList) - 1)[len(HeelList) - 1]  # 得到最大值的索引




        '''
        for i in range(len(HeelList) - MaxIndex - 11):  # 不会取到最后10个
            if (((HeelList[MaxIndex + i] <= Valve) and (
                    (HeelList[MaxIndex + i + 1] <= Valve))
                 and (abs(AccelerationList[MaxIndex + i]) < 100)
            )):
                return MaxIndex + i
        '''


        for i in range(len(HeelList) - MaxIndex - 11):  # 不会取到最后10个
            if ((HeelList[MaxIndex + i] <= HeelLiftValve + DyanmicParaValve) and (
                    (HeelList[MaxIndex + i + 1] <= HeelLiftValve + DyanmicParaValve))
                    and (abs(AccelerationList[MaxIndex + i + 1]) < 100)
                and (abs(AccelerationList[MaxIndex + i + 2]) < 100)
            ):
                return MaxIndex + i

        if(self.ErrorToChangeFlag(stepType)):
            print("HeelLift Error!!\t \t \t!!!!!!!!!!!!!!!!!--->", self.filepath,"--StartIndex：",StartIndex)
        return 40


    def __FindHeelLiftBySoleListOld(self, StartIndex, EndIndex,SoleArr,HeelArr,TorArr,stepType):

        SoleProcessArr = SoleArr[StartIndex - 20: EndIndex]
        SoleProcessFilterArr = np.array(Filter.SlideAvgfliter(SoleProcessArr, self.filterNumber))
        SoleFiltedArr = SoleProcessFilterArr[19:]
        SoleList = SoleFiltedArr[1:]
        SoleMaxIndex = np.argpartition(SoleList, len(SoleList) - 1)[len(SoleList) - 1]  # 得到最大值的索引

        HeelProcessArr = HeelArr[StartIndex - 20: EndIndex]
        HeelProcessFilterArr = np.array(Filter.SlideAvgfliter(HeelProcessArr, self.filterNumber))
        HeelFiltedArr = HeelProcessFilterArr[19:]
        AccelerationList = HeelFiltedArr[1:] - HeelFiltedArr[0:-1]  # 得到滤波后的加速度列表
        HeelList = HeelFiltedArr[1:]
        HeelMaxIndex = np.argpartition(HeelList, len(HeelList) - 1)[len(HeelList) - 1]  # 得到最大值的索引

        ToeProcessArr = TorArr[StartIndex - 20: EndIndex]
        ToeProcessFilterArr = np.array(Filter.SlideAvgfliter(ToeProcessArr, self.filterNumber))
        ToeFiltedArr = ToeProcessFilterArr[19:]
        ToeList = ToeFiltedArr[1:]
        ToeMaxIndex = np.argpartition(ToeList, len(ToeList) - 1)[len(ToeList) - 1]  # 得到最大值的索引
        if(ToeMaxIndex <= SoleMaxIndex): ToeMaxIndex = SoleMaxIndex + 5

        AccelerationValve = np.average(np.abs(AccelerationList[SoleMaxIndex + 1 : ToeMaxIndex + 1]))

        #if(SoleMaxIndex < ToeMaxIndex and SoleMaxIndex > HeelMaxIndex ) : return SoleMaxIndex


        for i in range(SoleMaxIndex + 1,ToeMaxIndex + 1):
            if(abs(AccelerationList[i]) <= AccelerationValve):
                return i


        if (self.ErrorToChangeFlag(stepType)):
            print("HeelLift Error!!\t \t \t!!!!!!!!!!!!!!!!!--->", self.filepath, "--StartIndex：", StartIndex)
        return 40

    def __FindSoleContactByAcceleration(self, StartIndex, EndIndex, Arr,stepType):
        try:
            # Arr = np.round(Arr)
            ProcessArr = Arr[StartIndex - 20: EndIndex]
            ProcessFilterArr = np.array(Filter.SlideAvgfliter(ProcessArr, self.filterNumber))
            FiltedArr = ProcessFilterArr[19:]
            CheckArr = FiltedArr[1:] - FiltedArr[0:-1]  # 得到滤波后的加速度列表
            SoleList = FiltedArr[1:]

            FirstFlag = False
            changeFirCondIndexFlag = False
            FirCondIndex = 0

            #SoleValve = np.max(ProcessFilterArr[:20])
            SoleValve = np.max(ProcessFilterArr[0:20])
            AccelerationValve = np.average(CheckArr[:10])

            MaxIndex = np.argpartition(SoleList, len(SoleList) - 1)[len(SoleList) - 1]  # 得到最大值的索引

            Valve =  max(SoleValve + self.DyanmicParaValve,SoleValve * self.multiple)


            for i in range(0, MaxIndex + 1):
                if (((CheckArr[MaxIndex - i + 1]) <= AccelerationValve + 20)) \
                        and (SoleList[MaxIndex - i] <= Valve):
                    return MaxIndex - i


            '''   2-26
            for i in range(0,MaxIndex + 1 ):
                if(((SoleList[MaxIndex - i] <= Valve ) or (CheckArr[MaxIndex - i]) <= AccelerationValve + 20))\
                        and (SoleList[MaxIndex - i - 1] <= Valve):
                    return MaxIndex - i
            '''
            '''  2-25
            for i in range(0,MaxIndex + 1 ):
                if(((SoleList[MaxIndex - i] <= SoleValve +self.DyanmicParaValve) or (CheckArr[MaxIndex - i]) <= AccelerationValve + 20))\
                        and (SoleList[MaxIndex - i - 1] <= SoleValve + self.DyanmicParaValve):
                    return MaxIndex - i
            '''

            if(self.ErrorToChangeFlag(stepType)):
                print("SoleError--> i ",MaxIndex - i,self.filepath,StartIndex)

            return 10
        except Exception:
            if (self.ErrorToChangeFlag(stepType)):
                print("SoleIndexError-->",self.filepath, StartIndex)

    # 获得全脚全相位FIT
    def GetTotalGlobalResultData(self):
        BAPResultData = FIT.ResultData(np.sum(self.BAPArr[:, 0:3], axis=1),self.Weight)  # 按行进行加法
        BTSResultData = FIT.ResultData(np.sum(self.BTSArr[:, 0:3], axis=1),self.Weight)
        APResultData = FIT.ResultData(np.sum(self.APArr[:, 0:3], axis=1),self.Weight)
        TSResultData = FIT.ResultData(np.sum(self.TSArr[:, 0:3], axis=1),self.Weight)
        DSResultData = FIT.ResultData(np.sum(self.DSArr[:, 0:3], axis=1),self.Weight)
        return BAPResultData, BTSResultData, APResultData, TSResultData, DSResultData

    # 获得全脚全相位FIT
    def GetGlobalResultData(self, Arr):
        '''
        :param Arr   BAP,AP,TS,DS n行4列矩阵:
        :return:
        '''
        Heel_GlobalResultData = FIT.ResultData(Arr[:, 3],self.Weight)
        Arch_GlobalResultData = FIT.ResultData(Arr[:, 2],self.Weight)
        Sole_GlobalResultData = FIT.ResultData(Arr[:, 1],self.Weight)
        Toe_GlobalResultData = FIT.ResultData(Arr[:, 0],self.Weight)
        Meta1_GlobalResultData = FIT.ResultData(Arr[:, 4],self.Weight)
        Meta2_GlobalResultData = FIT.ResultData(Arr[:, 5],self.Weight)
        Meta3_GlobalResultData = FIT.ResultData(Arr[:, 6],self.Weight)
        return Heel_GlobalResultData, Arch_GlobalResultData, Sole_GlobalResultData, Toe_GlobalResultData, \
               Meta1_GlobalResultData, Meta2_GlobalResultData, Meta3_GlobalResultData

    def GetPhaseResultData(self, PhaseIndexlist, Arr):
        '''
        :param HCIndexlist  HC阶段起始点位 :
        :param Arr BAP,AP,TS,DS n行4列矩阵:
        :return:
        '''
        Heel_ResultData = FIT.ResultData(Arr[PhaseIndexlist[0]:PhaseIndexlist[1], 3],self.Weight)
        Arch_ResultData = FIT.ResultData(Arr[PhaseIndexlist[0]:PhaseIndexlist[1], 2],self.Weight)
        Sole_ResultData = FIT.ResultData(Arr[PhaseIndexlist[0]:PhaseIndexlist[1], 1],self.Weight)
        Toe_ResultData = FIT.ResultData(Arr[PhaseIndexlist[0]:PhaseIndexlist[1], 0],self.Weight)
        Meta1_ResultData = FIT.ResultData(Arr[PhaseIndexlist[0]:PhaseIndexlist[1], 4],self.Weight)
        Meta2_ResultData = FIT.ResultData(Arr[PhaseIndexlist[0]:PhaseIndexlist[1], 5],self.Weight)
        Meta3_ResultData = FIT.ResultData(Arr[PhaseIndexlist[0]:PhaseIndexlist[1], 6],self.Weight)
        return Heel_ResultData, Arch_ResultData, Sole_ResultData, Toe_ResultData, \
               Meta1_ResultData, Meta2_ResultData, Meta3_ResultData

    def GetPeriod(self, HCList, MSList, TFList):
        HCPeriod = (HCList[1] - HCList[0]) / 100
        MSPeriod = (MSList[1] - MSList[0]) / 100
        TFPeriod = (TFList[1] - TFList[0]) / 100
        return HCPeriod, MSPeriod, TFPeriod


if __name__ == '__main__':
    filepath = "SumAreaData\\subject01\\2-2.csv"
    BAP = [452, 521]
    BTS = [508, 574]
    AP = [562, 630]
    TS = [618, 686]
    DS = [673, 742]
    A = ProcessData(filepath, BAP, BTS, AP, TS, DS)
