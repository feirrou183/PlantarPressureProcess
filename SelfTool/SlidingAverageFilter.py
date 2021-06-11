import csv
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import time


'''
这里假设采样频率为50hz,信号本身最大的频率为0.1hz，
要滤除0.1hz以上频率成分，即截至频率为2hz,则wn=2*0.2/50=0.04。Wn=0.02
'''

class Queue(object):
    """队列"""
    def __init__(self):
        self.items = []

    def is_empty(self):
        return self.items == []

    def enqueue(self, item):
        """进队列"""
        self.items.insert(0,item)

    def dequeue(self):
        """出队列"""
        return self.items.pop()

    def size(self):
        """返回大小"""
        return len(self.items)

def velocityfliter(dataList, WindowsLen):
    i = 0
    velocity = []
    cup = Queue()
    try:
        while True:
            if(cup.size() == WindowsLen):                   #滑动滤波，length个为1队列
                cup.dequeue()     #弹出最早的
                cup.enqueue(dataList[i])     #加入最新的
                velocity.append(sum(cup.items) / WindowsLen)         #放入数组
                i += 1
                continue
            else:                                          #不满足
                cup.enqueue(dataList[i])
                i +=1
                velocity.append(sum(cup.items)/cup.size())
    except IndexError:
        pass
    return velocity



if __name__ == "__main__":
    pass

    
        


