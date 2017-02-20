#!/usr/bin/env Python3
# -*- coding: utf-8 -*-

# @Name    : kNN
# @Author  : qianyong
# @Time    : 2017-02-20 17:09

"""
k-近邻算法
"""
from numpy import  *
import operator


class kNN(object):
    def __init__(self):
        pass

    def create_data_set(self):
        """
        创建数据集
        :return:
        """
        group = array([[1.0, 1.1],
                          [1, 1],
                          [0, 0],
                          [0, 0.1]])
        labels = ['A', 'A', 'B', 'B']
        return group, labels

    def class_if_y0(self, index, dataset, labels, k):
        """
        选择距离最少的 k 个点
        :param index: 分类的输入向量
        :param dataset:训练样本集
        :param labels:标签向量
        :param k:选择最近邻居的数目
        :return:
        """
        dataset_size = dataset.shape[0]              # shape 是获得 数据的 维度
        diff_mat = tile()


if __name__ == '__main__':
    demo = kNN()
    group, labels = demo.create_data_set()
    print(group)
    print(labels)
    # pass
