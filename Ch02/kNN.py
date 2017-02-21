#!/usr/bin/env Python3
# -*- coding: utf-8 -*-

# @Name    : kNN
# @Author  : qianyong
# @Time    : 2017-02-20 17:09

"""
k-近邻算法
"""
from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt


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

        # 计算距离
        dataset_size = dataset.shape[0]  # shape 是获得 数据的 维度
        diff_mat = tile(index, (dataset_size, 1)) - dataset  # tile  在列方向上重复index dataset_size 次，行1 次
        # 距离计算开始 (欧式距离)
        sq_diff_mat = diff_mat ** 2
        sq_distances = sq_diff_mat.sum(axis=1)
        distances = sq_distances ** 0.5
        # 距离计算结束

        sorted_distindicies = distances.argsort()  # argsort函数返回的是数组值从小到大的索引值
        class_count = {}
        # 计算最近的k 个点
        for i in range(k):
            vote_ilabel = labels[sorted_distindicies[i]]
            class_count[vote_ilabel] = class_count.get(vote_ilabel, 0) + 1
        # 类别降序
        sorted_distindicies = sorted(class_count.items(),
                                     key=operator.itemgetter(1), reverse=True)

        # 返回 类别
        return sorted_distindicies[0][0]

    def file2mat(self, filename):
        """
        将文件转换矩阵
        :param filename:
        :return:
        """

        # 打开文件
        fr = open(filename)
        # 得到文件行数
        number_of_lines = len(fr.readlines())
        # 填充矩阵
        return_mat = zeros((number_of_lines, 3))
        class_label_vector = []
        fr = open(filename)
        index = 0

        # 标签类别
        labels = {}
        labels_num = 1
        for line in fr.readlines():
            line = line.strip()
            list_from_line = line.split('\t')
            return_mat[index, :] = list_from_line[0:3]
            if not labels.get(list_from_line[-1]):
                labels[list_from_line[-1]] = labels_num
                labels_num += 1

            class_label_vector.append(labels.get(list_from_line[-1]))
            index += 1
        print('标签分类为：', labels)
        return return_mat, class_label_vector

    def mat2pic(self, dataset, data_labels):
        """
        使用matplotlib  画出散点图
        :param dataset:
        :return:
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # ax.scatter(dataset[:, 1], dataset[:, 2], 15.0 * array(data_labels),15.0 * array(data_labels))
        # ax1 = fig.add_subplot(2,2,2)
        # ax1.scatter(dataset[:, 0], dataset[:, 2], 15.0 * array(data_labels), 15.0 * array(data_labels))
        # ax2 = fig.add_subplot(2, 2, 3)
        ax.scatter(dataset[:, 0], dataset[:, 1], 15.0 * array(data_labels), 15.0 * array(data_labels))
        plt.show()

    def auto_norm(self, dataset):
        """
        归一化特征值
        :param dataset:
        :return:
        """
        # 0 是列， 1 是行
        min_vals = dataset.min(0)
        max_vals = dataset.max(0)
        ranges = max_vals - min_vals
        # norm_set = zeros(shape(dataset))
        m = dataset.shape[0]

        norm_set = dataset - tile(min_vals, (m, 1))
        norm_set = norm_set / tile(ranges, (m, 1))  # 特征值相除
        return norm_set, ranges, min_vals

    def data_class_test(self, filename):
        """
        测试分类器效果
        :return:
        """
        # 测试 比例
        test_ratio = 0.2
        group, labels = self.file2mat(filename)
        norm_mat, ranges, min_vals = self.auto_norm(group)
        m = norm_mat.shape[0]
        numtest_vecs = int(m * test_ratio)
        error_count = 0
        for i in range(numtest_vecs):
            # print(norm_mat[numtest_vecs:m, :])
            class_if_result = self.class_if_y0(norm_mat[i, :],
                                               norm_mat[numtest_vecs:m, :],
                                               labels[numtest_vecs:m], 26)
            print('the calssifer came back with: {},the real answer is:{}'.format(
                class_if_result, labels[i]))
            if class_if_result != labels[i]:
                # print('-----------------------')
                error_count += 1
        print('the totel error rate is :{}'.format(float(error_count/numtest_vecs)),error_count)

if __name__ == '__main__':
    demo = kNN()
    filename = r'C:\Users\wqy\Desktop\机器学习实战pdf+源码\MLiA_SourceCode\machinelearninginaction\Ch02\datingTestSet.txt'
    demo.data_class_test(filename)
    # group, labels = demo.file2mat(filename)
    # norm_mat, ranges, min_vals = demo.auto_norm(group)
    # print(norm_mat)
    # print(ranges)
    # print(min_vals)
    # demo.mat2pic(norm_mat, labels)
