#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/5/6 21:19
# @Author : Yaokang Zhu
# @Site : 
# @File : graphVec.py
# @Software: PyCharm
import numpy as np


def graphVec(node_feat,adj_one,adj_one_test):
    count = 0
    node_feat = np.array(node_feat)
    node_feat_new0 = np.zeros((122747, 37))
    node_feat_new1 = np.zeros((122747, 74))
 

    q = 0
    print("Initialize..")
    for i in range(len(adj_one)):
    # for i in range(2):
        node_feat_subSUM = np.zeros((1, 37))
        for j in range(len(adj_one[i])):
            node_feat_subSUM = node_feat[count + j] + node_feat_subSUM
            node_feat_sum = np.zeros(37)
            node_feat_sum2 = np.zeros(37)
            node_feat_sum3 = np.zeros(37)
            sum_neighbour2 = 0
            sum_neighbour3 = 0
            for k in range(len(adj_one[i][j])):

                """
                               1hop 2hop 3hop neighbours
                                """
                node_feat_sum = node_feat[count + adj_one[i][j][k]] + node_feat_sum
                for k2 in range(len(adj_one[i][adj_one[i][j][k]])):
                    node_feat_sum2 = node_feat[count + adj_one[i][adj_one[i][j][k]][k2]] + node_feat_sum2
                    sum_neighbour2 = sum_neighbour2 + 0.5
                    for k3 in range(len(adj_one[i][adj_one[i][adj_one[i][j][k]][k2]])):

                        node_feat_sum3 = node_feat[
                                             count + adj_one[i][adj_one[i][adj_one[i][j][k]][k2]][k3]] + node_feat_sum3
                        sum_neighbour3 = sum_neighbour3 + 0.3
            # nodefeat_3order = node_feat_sum3
            nodefeat_3order = node_feat_sum3
            node_feat_new0[q + j] = node_feat[count + j]
            # nodefeat_1order = ( nodefeat_3order )
            nodefeat_1order = (node_feat_sum3+ node_feat_sum2)
            node_feat_new1[q + j] = np.append(node_feat_new0[q + j], nodefeat_1order)
        count = count + len(adj_one[i])
        q = q + len(adj_one[i])

    q350 = 0
    count350 = 0


    for i in range(len(adj_one_test)):
        node_feat_subSUM = np.zeros((1, 37))
        for j in range(len(adj_one_test[i])):
            node_feat_subSUM = node_feat[count + count350 + j] + node_feat_subSUM
            node_feat_sum = np.zeros(37)
            node_feat_sum2 = np.zeros(37)
            node_feat_sum3 = np.zeros(37)
            sum_neighbour2 = 0
            sum_neighbour3 = 0
            for k in range(len(adj_one_test[i][j])):
                """
               1hop 2hop 3hop neighbours
                """
                node_feat_sum = node_feat[count + count350 + adj_one_test[i][j][k]] + node_feat_sum
                for k2 in range(len(adj_one_test[i][adj_one_test[i][j][k]])):
                    node_feat_sum2 = node_feat[
                                         count + count350 + adj_one_test[i][adj_one_test[i][j][k]][k2]] + node_feat_sum2
                    sum_neighbour2 = sum_neighbour2 + 0.5
                    for k3 in range(len(adj_one_test[i][adj_one_test[i][adj_one_test[i][j][k]][k2]])):
                        node_feat_sum3 = node_feat[
                                             count + count350 +
                                             adj_one_test[i][adj_one_test[i][adj_one_test[i][j][k]][k2]][
                                                 k3]] + node_feat_sum3
                        sum_neighbour3 = sum_neighbour3 + 0.3
            nodefeat_3order = node_feat_sum3
            node_feat_new0[count + q350 + j] = node_feat[count + count350 + j]
            # nodefeat_1order = (nodefeat_3order )
            nodefeat_1order = (node_feat_sum3+node_feat_sum2)
            node_feat_new1[count + q350 + j] = np.append(node_feat_new0[count + q350 + j], nodefeat_1order)
        count350 = count350 + len(adj_one_test[i])
        q350 = q350 + len(adj_one_test[i])
    return node_feat_new1