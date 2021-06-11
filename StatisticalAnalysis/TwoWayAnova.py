import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
#用于Len
import scipy.stats as stats
import scipy.optimize as opt

import os
Work_Path = "F:\\PlantarPressurePredictExperiment"
FileName = "OutPut\\TwoWayAnovaData\\TwoWayAnovaData.xlsx"
os.chdir(Work_Path)

#进行双因素anova分析。  分析角度，策略对值的影响,其中data为pandas的标准列表。
def TwoWayAnonva(data,objName):
    model = ols('{0}~C(angle) + C(strategy) + C(angle):C(strategy)'.format(objName), data).fit()
    ans = anova_lm(model)
    ansLine = [ans['PR(>F)']['C(angle)'],ans['PR(>F)']['C(strategy)'],ans['PR(>F)']['C(angle):C(strategy)']]
    return ansLine

def Levenes(first,second):
    res = stats.levene(first,second)
    return res

#绘制关联矩阵
def drawFactorMatrix():
    pass

def writeAnsCsv():
    pass



if __name__ == '__main__':
    #以第一行作为行名
    data = pd.read_excel(FileName,sheet_name=0,header= 0)











