import os
import re

classify = ['1','2','3','4','5','6','7','8','9','10','11']
item = ['1','2','3','4','5','6','7','8','9','10']
foot = ["L","R"]

os.chdir("/")

fileNumber = 76
index = 0
subject = "subject12"
filepath = "RawData\\" + subject
fileList = os.listdir(filepath)


def ChangeName():
    index = 0
    for i in range(len(classify)):
        for j in range(len(item)):
            for k in range(len(foot)):
                if(i<=4):   #前5项
                    os.rename(filepath + "\\"+fileList[index], filepath + "\\"+"{0}-{1}-{2}.csv".format(classify[i],item[j],foot[k]))
                    index += 1
                elif(i<=6): #转圈
                    if(j <= 2): #3个
                        os.rename(filepath + "\\" + fileList[index], filepath + "\\" +"{0}-{1}-{2}.csv".format(classify[i], item[j], foot[k]))
                        index += 1
                elif(i<=10):
                    if(j <= 4):  #前5个
                        os.rename(filepath + "\\" + fileList[index], filepath + "\\" +"{0}-{1}-{2}.csv".format(classify[i], item[j], foot[k]))
                        index += 1


def ChangeVedioName():
    index = 0
    Work_Path = "/Vedio"
    os.chdir(Work_Path)
    fileList = os.listdir(subject)
    for i in range(len(classify)):
        for j in range(len(item)):
            if(i<=4):   #前5项
                os.rename(subject + "\\"+fileList[index], subject + "\\"+"{0}-{1}.mp4".format(classify[i],item[j]))
                index += 1
            elif(i<=6): #转圈
                if(j <= 2): #3个
                    os.rename(subject + "\\" + fileList[index], subject + "\\" +"{0}-{1}.mp4".format(classify[i],item[j]))
                    index += 1
            elif(i<=10):
                if(j <= 4):  #前5个
                    os.rename(subject + "\\" + fileList[index], subject + "\\" +"{0}-{1}.mp4".format(classify[i],item[j]))
                    index += 1
                
            
            
    
    




def Check(fileList):
    num = range(1,78)   #项目总数
    dic = {}
    for i in num:
        dic[str(i).zfill(2)] = 0    #初始化字典
    for j in fileList:
        key = re.split('ZengY|L|R',j)[1]     #分割出来是这样   ['', '01', '_M.csv']
        dic[key] +=1

    for item in dic:
        if(dic[item] != 2) :
            print(item)



if __name__ == '__main__':
    #Check(fileList)   #检查数量是否够
    #ChangeName()    #原文件改名
    ChangeVedioName()
    


        

            
        

