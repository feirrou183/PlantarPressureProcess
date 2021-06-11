
import os
Work_Path = "F:\\PlantarPressurePredictExperiment"
os.chdir(Work_Path)

import csv
import numpy as np
import ProcessProgram.ProcessSumAreaData as Pro

def Process(subject,item1,FileList):
    #item1 = ["2","3","4","5"]    #角度大类
    item2 = ["1","2","3","4","5","6","7","8","9","10","11"]   #每个角度小类
    WeigithList = [64.8,70.55,54.55,56.65,71.7,67.85,70.05,66.25,64.35,66.05,65.8,61.65]
    Weight = WeigithList[int(subject)-1]

    X = ["Heel","Arch","Sole","Toe","Meta1","Meta2","Meta3","Total"]
    Y = ["BAP","BTS","AP","TS","DS"]
    Z = ["Global","HC","MS","TF"]

    Dic = {}
    i = 0
    for j in range(len(item1)):
        for k in range(len(item2)):
            filePath = "SumAreaData\\subject{}\\{}-{}.csv".format(subject,item1[j],item2[k])
            if(os.path.exists(filePath) == False):
                if(i+5 < len(FileList)-1 and FileList[i] == ""): i += 5
                continue
            BAP = FileList[i]
            BTS = FileList[i+ 1]
            AP = FileList[i + 2]
            TS = FileList[i + 3]
            DS = FileList[i + 4]
            itemIndex = "{}-{}".format(item1[j], item2[k])
            Dic[itemIndex] = {}
            Dic[itemIndex]["BAP"] = BAP.replace('-',',')
            Dic[itemIndex]["BTS"] = BTS.replace('-', ',')
            Dic[itemIndex]["AP"] = AP.replace('-',',')
            Dic[itemIndex]["TS"] = TS.replace('-',',')
            Dic[itemIndex]["DS"] = DS.replace('-',',')
            i += 5
            step = eval("[[{0}],[{1}],[{2}],[{3}],[{4}]]".format(Dic[itemIndex]["BAP"],Dic[itemIndex]["BTS"],Dic[itemIndex]["AP"],Dic[itemIndex]["TS"],Dic[itemIndex]["DS"]))
            ProcessData = Pro.ProcessData(filePath,step[0],step[1],step[2],step[3],step[4],Weight)

            Dic[itemIndex]["ErrorFlag"] = ProcessData.ErrorFlag

            for l in Y:
                Dic[itemIndex]["{0}Period".format(l)] = eval("ProcessData.{0}Period".format(l))

            for o in Y:
                for p in ["SCValue","HLValue"]:
                    Dic[itemIndex]["{0}{1}".format(o, p)] = eval("ProcessData.{0}{1}".format(o, p))

            for r in Y:
                Dic[itemIndex]["{}ErrorFlag".format(r)] = eval("ProcessData.{0}ErrorFlag".format(r))

            for m in Y:
                for n in ["HC","MS","TF"]:
                    PeriodName = "{0}_{1}Period".format(m,n)
                    Dic[itemIndex][PeriodName] = eval("ProcessData."+PeriodName)
            print("Processint",filePath,"DS--->",str(DS),"HCPeriod--->",ProcessData.AP_HCPeriod)
            for a in X:
                for b in Y:
                    for c in Z:
                        itemName = "{0}_{1}_{2}ResultData".format(a,b,c)
                        if(hasattr(ProcessData,itemName)):
                            Dic[itemIndex][itemName] = \
                                {"FTI": eval("ProcessData."+itemName +".FTI"), "aMax" : eval("ProcessData." + itemName + ".aMax")}
    return Dic



if __name__ == '__main__':
    Dic = Process()