import os
import json
import csv
import numpy as np

subjectNumbers = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", ]
subjects = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", ]
items = ["1", "2", "3", "4", "5"]
sub_items = ["1", "2", "3", "4", "5","6", "7","8", "9", "10","11"]

Work_Path = "F:\\PlantarPressurePredictExperiment"
os.chdir(Work_Path)

import ProcessProgram.TempProgram.ProcessIndexListToValue as PTL



class DicManager:
    def __init__(self):
        self.subjectConfigDict = self.getDic()
        self.FirstInit()

    def getDic(self):
        if (os.path.exists("subjectConfig.json") == False):
            diction = {}
            with open("subjectConfig.json", "w") as f:
                json.dump(diction, f)
                print("预创建成功")

        with open("subjectConfig.json", "r") as f:
            subjectConfigDict = json.load(f)
        return  subjectConfigDict

    def FirstInit(self):
        for i in range(len(subjectNumbers)):
            self.subjectConfigDict.setdefault("subject{}".format(subjectNumbers[i]),{})
            #self.subjectConfigDict["subject{}".format(subjectNumbers[i])] = {}

    def addValue(self,subject,key,value):
        if(subject in self.subjectConfigDict):
            self.subjectConfigDict[subject][key] = value
        else:
            self.subjectConfigDict[subject] = {}
            self.subjectConfigDict[subject][key] = value

    def addKeyValue(self,subject,key1,key2,value):
        '''
        三层嵌套字典赋值。
        :param subject:
        :param key1:
        :param key2:
        :param value:
        :return:
        '''
        if (subject in self.subjectConfigDict):
            if (key1 in self.subjectConfigDict[subject]):
                self.subjectConfigDict[subject][key1][key2] = value
                return

        print("Wrong,key were Can Not Be Found!")

    def addKeyKeyValue(self,subject,key1,key2,key3,value):
        if (subject in self.subjectConfigDict):
            if (key1 in self.subjectConfigDict[subject]):
                if(key2 in self.subjectConfigDict[subject][key1]):
                    self.subjectConfigDict[subject][key1][key2][key3] = value
                    return
        print("Wrong,key were Can Not Be Found!")

    def additemDict(self,item):
        '''
        为所有subject添加一项新字典。
        :return:
        '''
        for i in self.subjectConfigDict:
            self.subjectConfigDict[i][item] = {}

    def DictDump(self):
        with open("subjectConfig.json","w",encoding="utf-8") as f:
            json.dump(self.subjectConfigDict,f,indent=2,sort_keys=True,ensure_ascii=False)


class DataExport:
    def __init__(self,dic,filePath):
        self.dic = dic
        self.filePath = filePath
        self.OutputArr = self.GetOutPutType()


    def WriteCsv(self):
        X = ["Heel","Arch","Sole","Toe","Meta1","Meta2","Meta3","Total"]
        Y = ["BAP","BTS", "AP", "TS", "DS"]
        Z = ["Global", "HC", "MS", "TF"]


        with open(self.filePath, 'a',newline='') as f:
            writer = csv.writer(f)
            firstRow = []
            firstRow.append("subject")
            firstRow.append("ItemNumber")
            for r in Y:
                firstRow.append("{}ErrorFlag".format(r))
            firstRow.append("BAP")
            firstRow.append("BTS")
            firstRow.append("AP")
            firstRow.append("TS")
            firstRow.append("DS")

            for e in ["BAP","BTS","AP","TS","DS"]:
                for f in ["SCValue","HLValue"]:
                    firstRow.append("{0}{1}".format(e,f))

            for m in ["BAP","BTS", "AP", "TS", "DS"]:
                for n in ["HC", "MS", "TF"]:
                    PeriodName = "{0}_{1}Period".format(m, n)
                    firstRow.append(PeriodName)


            for a in X:
                for b in Y:
                    for c in Z:
                        detailItemName = "{0}_{1}_{2}ResultData".format(a, b, c)
                        if (self.dic["subject01"]["ResultData"]["2-3"].__contains__(detailItemName)):
                            firstRow.append(detailItemName + "FTI")
                            firstRow.append(detailItemName + "aMax")
            writer.writerow(firstRow)
            for i in self.OutputArr:
                writer.writerow(i)






    def GetOutPutType(self):
        subject = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
        item1 = ["1","2", "3", "4", "5"]  # 角度大类
        item2 = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]  # 每个角度小类

        X = ["Heel","Arch","Sole","Toe","Meta1","Meta2","Meta3","Total"]
        Y = ["BAP","BTS", "AP", "TS", "DS"]
        Z = ["Global", "HC", "MS", "TF"]
        Arr = []
        for i in range(len(subject)):
            for j in range(len(item1)):
                for k in range(len(item2)):
                    line = []
                    subjectNum = "subject" + str(subject[i])
                    itemNum = "{0}-{1}".format(item1[j],item2[k])
                    line.append(subjectNum)    #subject
                    line.append(itemNum)       #itemNum
                    if((self.dic[subjectNum]["ResultData"].__contains__(itemNum)) == False):  continue   #不存在此文件
                    for r in Y:
                        line.append(self.dic[subjectNum]["ResultData"][itemNum]["{}ErrorFlag".format(r)])
                    line.append(self.dic[subjectNum]["ResultData"][itemNum]["BAP"])
                    line.append(self.dic[subjectNum]["ResultData"][itemNum]["BTS"])
                    line.append(self.dic[subjectNum]["ResultData"][itemNum]["AP"])
                    line.append(self.dic[subjectNum]["ResultData"][itemNum]["TS"])
                    line.append(self.dic[subjectNum]["ResultData"][itemNum]["DS"])

                    for e in ["BAP", "BTS", "AP", "TS", "DS"]:
                        for f in ["SCValue", "HLValue"]:
                            line.append(self.dic[subjectNum]["ResultData"][itemNum]["{0}{1}".format(e,f)])

                    for m in Y:
                        for n in ["HC", "MS", "TF"]:
                            PeriodName = "{0}_{1}Period".format(m, n)
                            line.append(self.dic[subjectNum]["ResultData"][itemNum][PeriodName])

                    for a in X:
                        for b in Y:
                            for c in Z:
                                detailItemName = "{0}_{1}_{2}ResultData".format(a, b, c)
                                if (self.dic[subjectNum]["ResultData"][itemNum].__contains__(detailItemName)):
                                    if(self.dic[subjectNum]["ResultData"][itemNum]["{}ErrorFlag".format(b)] == True):
                                        line.append(" ")
                                        line.append(" ")
                                        continue
                                    line.append(self.dic[subjectNum]["ResultData"][itemNum][detailItemName]["FTI"])
                                    line.append(self.dic[subjectNum]["ResultData"][itemNum][detailItemName]["aMax"])
                    Arr.append(line)
                    print(subjectNum + "----->" + itemNum)
        return np.array(Arr)


def GetSubjectDict():
    subjectDic = {}
    subjectIndex = ["01","02","03","04","05","06","07","08","09","10","11","12"]
    subjectItem1Index = ["1","2","3","4","5"]

    subjectDic[subjectIndex[0]] = subjectItem1Index      # subject01
    subjectDic[subjectIndex[1]] = subjectItem1Index      # subject02
    subjectDic[subjectIndex[2]] = subjectItem1Index      # subject03
    subjectDic[subjectIndex[3]] = subjectItem1Index      # subject04
    subjectDic[subjectIndex[4]] = subjectItem1Index      # subject05

    subjectDic[subjectIndex[5]] = subjectItem1Index      # subject06
    subjectDic[subjectIndex[6]] = subjectItem1Index      # subject07
    subjectDic[subjectIndex[7]] = subjectItem1Index      # subject08
    subjectDic[subjectIndex[8]] = subjectItem1Index      # subject09
    subjectDic[subjectIndex[9]] = subjectItem1Index      # subject010

    subjectDic[subjectIndex[10]] = subjectItem1Index    # subject11
    subjectDic[subjectIndex[11]] = subjectItem1Index    # subject12
    return subjectDic

def GetList(Arr,i):
    list = Arr[:, int(i) - 1].tolist()
    try:
        while (1):
            list.remove('')
    except ValueError:
        pass
    return list


class GetCsvFile:
    def __init__(self,filePath):
        self.file = filePath


    def GetArr(self):
        with open(self.file) as f:
            reader = csv.reader(f)
            rows = [row for row in reader]
        Arr = np.array(rows)
        return Arr
        

def DumpLabel(dicManager):
    labelArr = np.loadtxt("labels.csv", delimiter=",", dtype=np.str)
    i = 0
    k = 0

    for subject in subjects:
        for item in items:
            for sub_item in sub_items:
                detailItemName = "{0}-{1}".format(item, sub_item)
                subjectName = "subject{}".format(subject)
                if (dicManager.subjectConfigDict[subjectName]["ResultData"].__contains__(detailItemName)):
                    dicManager.subjectConfigDict[subjectName]["ResultData"][detailItemName]["angle"] = labelArr[i, 0]
                    dicManager.subjectConfigDict[subjectName]["ResultData"][detailItemName]["strategy"] = labelArr[i, 1]
                    i += 1
                else:
                    print(subjectName, detailItemName)




if __name__ == '__main__':
    dicManager = DicManager()




    SubjectDic = GetSubjectDict()
    Arr = GetCsvFile("TempFile.csv").GetArr()


    for i in SubjectDic.keys():
        filelist = GetList(Arr,i)
        subjectStepDic = PTL.Process(i,SubjectDic[i],filelist)
        #dicManager.subjectConfigDict["subject{0}".format(i)]["ResultData"] = subjectStepDic
        dicManager.subjectConfigDict["subject{0}".format(i)]["ResultData"].update(subjectStepDic)

    dicManager.DictDump()

    Export = DataExport(dicManager.subjectConfigDict,"OutPut\\OutPutValue6-11.csv")
    Export.WriteCsv()

    pass















    






            
        

