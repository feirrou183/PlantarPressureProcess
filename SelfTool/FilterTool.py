import csv
import matplotlib.pyplot as plt
import numpy as np

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

def SlideAvgfliter(velocity_0,length):
    i = 0
    velocity = []
    cup = Queue()
    try:
        while True:
            if(cup.size() == length):                   #滑动滤波，length个为1队列
                cup.dequeue()     #弹出最早的
                cup.enqueue(velocity_0[i])     #加入最新的
                velocity.append(sum(cup.items)/cup.size())         #放入数组
                i += 1
                continue
            else:                                          #不满足
                cup.enqueue(velocity_0[i])
                i +=1
                velocity.append(sum(cup.items)/cup.size())
    except IndexError:
        pass
    return velocity

def MiddleFiliter(velocity_0,length):
     if(length%2 != 1) : return velocity_0   #确保length为奇数
     i = 0
     velocity = []
     wing = length // 2
     for j in range(length):
         velocity.append(velocity_0[j])
         i += 1
     pass



