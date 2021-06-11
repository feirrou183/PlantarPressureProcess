
#输入压力列表，获取FTI,平均压力峰值aMax
class ResultData:
    def __init__(self,PressureList,weight):
        self.PressureList = PressureList
        self.Weight = weight
        self.NewtonList = self.GetNewtonList()
        #压力时间积分
        self.FTI = self.GetFIT()
        #平均压力峰值
        self.aMax = self.GetAverageMax()

    '''
    def GetNewton(self, value):
        # 保留两位小数
        #2.58064E-2 = ka -> Pa     1e3  * 2.58064*E-5
        return round(value*2.58064E-2,3)
    '''

    def GetNewton(self, value):
        # 保留两位小数,加入体重参数影响。
        #2.58064E-2 = ka -> Pa     1e3  * 2.58064*E-5
        return round(value*2.58064E-2/self.Weight,3)

    def GetNewtonList(self):
        a = []
        for i in self.PressureList:
            a.append(self.GetNewton(i))
        return a

    def GetFIT(self):
        '''
        获取压力时间积分
        :return:
        '''
        FTI = 0
        for i in self.NewtonList:
            FTI += i
        return round(FTI/100,2)  #求出压力时间积分，保留两位小数

    def GetAverageMax(self):
        '''
        获取平均压力峰值
        :return:
        '''
        if(len(self.NewtonList) ==0): return 0

        return max(self.NewtonList)




if __name__ == '__main__':
    import os

    # 设置工作环境
    Work_Path = "F:\\PlantarPressurePredictExperiment"
    os.chdir(Work_Path)
    pass


