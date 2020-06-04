#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/5/5 22:47
# @Author : Avigdor
# @Site :
# @File : graphVec.py
# @Software: PyCharm

import numpy as np
import torch

def graphVec(node_feat,adj_one,adj_one_test):
    count = 0
    node_feat = np.array(node_feat)
    node_feat_new0 = np.zeros((8792, 19))
    node_feat_new1 = np.zeros((8792, 38))
    node_feat_new2 = np.zeros((8792, 57))

    node_feat_new3 = np.zeros((8792, 76))
    ##If you want to aggregate neighbors, you can use the code of this file
    q = 0
    print("Initialize..")
    for i in range(len(adj_one)):
        node_feat_subSUM = np.zeros((1, 19))
        for j in range(len(adj_one[i])):
            node_feat_subSUM = node_feat[count + j] + node_feat_subSUM
            node_feat_sum = np.zeros(19)
            node_feat_sum2 = np.zeros(19)
            node_feat_sum3 = np.zeros(19)
            sum_neighbour = 0
            sum_neighbour2 = 0
            sum_neighbour3 = 0
            for k in range(len(adj_one[i][j])):
                node_feat_sum = node_feat[count + adj_one[i][j][k]] + node_feat_sum
                for k2 in range(len(adj_one[i][adj_one[i][j][k]])):

                    node_feat_sum2 = node_feat[count + adj_one[i][adj_one[i][j][k]][k2]] + node_feat_sum2
                    sum_neighbour2 = sum_neighbour2 + 1

                    for k3 in range(len(adj_one[i][adj_one[i][adj_one[i][j][k]][k2]])):

                        node_feat_sum3 = node_feat[
                                             count + adj_one[i][adj_one[i][adj_one[i][j][k]][k2]][k3]] + node_feat_sum3

                        sum_neighbour3 = sum_neighbour3 + 1

                sum_neighbour = sum_neighbour + 1

            nodefeat_2order = node_feat_sum2
            nodefeat_1order = node_feat_sum
            # nodefeat_3order = node_feat_sum3
            #####################################
            # nodefeat_3order = node_feat_sum3
            ###################################################
            nodefeat_3order = node_feat_sum3 + node_feat_sum+node_feat_sum2
            ### Generate LA

            # if i==0 and j==0:
            #     nodefeat_2order_all = node_feat_sum2
            #     nodefeat_1order_all = node_feat_sum
            #     nodefeat_3order_all = node_feat_sum3
            #
            #     nodefeat_2order_all = nodefeat_2order_all.reshape((1,19))
            #     nodefeat_1order_all = nodefeat_1order_all.reshape((1,19))
            #     nodefeat_3order_all = nodefeat_3order_all.reshape((1,19))
            #
            # else:
            #     nodefeat_2order_all =  np.row_stack((nodefeat_2order_all,node_feat_sum2))
            #     nodefeat_1order_all =np.row_stack((nodefeat_1order_all,node_feat_sum))
            #     nodefeat_3order_all =np.row_stack((nodefeat_3order_all,node_feat_sum3))


            ###########################################################################
            node_feat_new0[q + j] = node_feat[count + j]
            nodefeat_1order = 90 * (nodefeat_3order)
            nodefeat_2order = 90 * (nodefeat_2order)
            nodefeat_3order = 90 * (
                nodefeat_3order)
            node_feat_new1[q + j] = np.append(node_feat_new0[q + j], nodefeat_1order)
            node_feat_new2[q + j] = np.append(node_feat_new1[q + j], nodefeat_2order)
            node_feat_new3[q + j] = np.append(node_feat_new2[q + j], nodefeat_3order)

        count = count + len(adj_one[i])
        q = q + len(adj_one[i])

    q350 = 0
    count350 = 0
    for i in range(len(adj_one_test)):
        node_feat_subSUM = np.zeros((1, 19))
        for j in range(len(adj_one_test[i])):
            node_feat_subSUM = node_feat[count + count350 + j] + node_feat_subSUM
            node_feat_sum = np.zeros(19)
            node_feat_sum2 = np.zeros(19)
            node_feat_sum3 = np.zeros(19)
            sum_neighbour = 0
            sum_neighbour2 = 0
            sum_neighbour3 = 0
            for k in range(len(adj_one_test[i][j])):
                """
                将每个一阶邻居求和得到平均值
                """
                node_feat_sum = node_feat[count + count350 + adj_one_test[i][j][k]] + node_feat_sum
                for k2 in range(len(adj_one_test[i][adj_one_test[i][j][k]])):
                    node_feat_sum2 = node_feat[
                                         count + count350 + adj_one_test[i][adj_one_test[i][j][k]][k2]] + node_feat_sum2
                    sum_neighbour2 = sum_neighbour2 + 1
                    for k3 in range(len(adj_one_test[i][adj_one_test[i][adj_one_test[i][j][k]][k2]])):
                        node_feat_sum3 = node_feat[
                                             count + count350 +
                                             adj_one_test[i][adj_one_test[i][adj_one_test[i][j][k]][k2]][
                                                 k3]] + node_feat_sum3
                        sum_neighbour3 = sum_neighbour3 +1
                sum_neighbour = sum_neighbour + 1
            nodefeat_2order = node_feat_sum2
            nodefeat_1order = node_feat_sum
            # nodefeat_2order_all = np.row_stack((nodefeat_2order_all, node_feat_sum2))
            # nodefeat_1order_all = np.row_stack((nodefeat_1order_all, node_feat_sum))
            # nodefeat_3order_all = np.row_stack((nodefeat_3order_all, node_feat_sum3))

            #####################################
            # nodefeat_3order = node_feat_sum3
            ###################################################
            nodefeat_3order = node_feat_sum3+ node_feat_sum2+node_feat_sum
            node_feat_new0[count + q350 + j] = node_feat[count + count350 + j]

            #################################################################
            nodefeat_1order = 90 * (nodefeat_3order)
            nodefeat_2order = 90 * (nodefeat_2order)
            nodefeat_3order = 90 * (
                nodefeat_3order)
            node_feat_new1[count + q350 + j] = np.append(node_feat_new0[count + q350 + j], nodefeat_1order)
            node_feat_new2[count + q350 + j] = np.append(node_feat_new1[count + q350 + j], nodefeat_2order)
            node_feat_new3[count + q350 + j] = np.append(node_feat_new2[count + q350 + j], nodefeat_3order)

            ###########################################################################

        count350 = count350 + len(adj_one_test[i])
        q350 = q350 + len(adj_one_test[i])
    # import pickle
    # print("nodefeat_2order_all",nodefeat_2order_all.shape)
    # fw2 = open('2order_LA_PTC.pkl', 'wb')
    # pickle.dump(nodefeat_2order_all, fw2)
    # fw1 = open('1order_LA_PTC.pkl', 'wb')
    # pickle.dump(nodefeat_1order_all, fw1)
    # fw3 = open('3order_LA_PTC.pkl', 'wb')
    # pickle.dump(nodefeat_3order_all, fw3)

    return node_feat_new1