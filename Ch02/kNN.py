#!/usr/bin/env Python3
# -*- coding: utf-8 -*-

# @Name    : kNN
# @Author  : qianyong
# @Time    : 2017-02-20 17:09

"""
k-近邻算法
"""

import numpy as np
import operator

class kNN(object):
    def __init__(self):
        pass

    def create_data_set(self):
        """
        创建数据集
        :return:
        """
        group = np.array([[1.0, 1.1],
                          [1, 1],
                          [0, 0],
                          [0, 0.1]])
        labels = ['A', 'A', 'B', 'B']
        return group, labels


if __name__ == '__main__':
    demo = kNN()
    group, labels = demo.create_data_set()
    print(group)
    print(labels)
    # pass
