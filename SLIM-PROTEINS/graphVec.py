#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/5/4 15:48
# @Author :Avigdor
# @Site : 
# @File : graphVec.py
# @Software: PyCharm
import numpy as np

def graphVec(node_feat,adj_one,adj_one_test):
    count = 0
    node_feat1 = node_feat
    node_feat1 = np.array(node_feat1)
    node_feat = np.array(node_feat)

    node_feat_new0 = np.zeros((43471, 3))
    node_feat_new1 = np.zeros((43471, 6))
    node_feat_new2 = np.zeros((43471, 9))
    ##If you want to aggregate neighbors, you can use the code of this file
    q = 0
    print("Initialize..")

    for i in range(len(adj_one)):

        node_feat_subSUM = np.zeros((1, 3))
        print("i",i)
        for j in range(len(adj_one[i])):
            node_feat_subSUM = node_feat[count + j] + node_feat_subSUM

            node_feat_sum = np.zeros(3)
            node_feat_sum2 = np.zeros(3)
            node_feat_sum3 = np.zeros(3)
            node_feat_sum4 = np.zeros(3)
            node_feat_sum5 = np.zeros(3)

            sum_neighbour = 0
            sum_neighbour2 = 0
            sum_neighbour3 = 0
            sum_neighbour4 = 0
            sum_neighbour5 = 0

            for k in range(len(adj_one[i][j])):


                node_feat_sum = node_feat[count + adj_one[i][j][k]] + node_feat_sum

                for k2 in range(len(adj_one[i][adj_one[i][j][k]])):

                    node_feat_sum2 = node_feat[count + adj_one[i][adj_one[i][j][k]][k2]] + node_feat_sum2

                    sum_neighbour2 = sum_neighbour2 + 1

                    for k3 in range(len(adj_one[i][adj_one[i][adj_one[i][j][k]][k2]])):

                        node_feat_sum3 = node_feat[
                                             count + adj_one[i][adj_one[i][adj_one[i][j][k]][k2]][k3]] + node_feat_sum3

                        sum_neighbour3 = sum_neighbour3 + 1
                        for k4 in range(len(adj_one[i][adj_one[i][adj_one[i][adj_one[i][j][k]][k2]][k3]])):

                            node_feat_sum4 = node_feat[
                                                 count + adj_one[i][adj_one[i][adj_one[i][adj_one[i][j][k]][k2]][k3]][
                                                     k4]] + node_feat_sum4
                            sum_neighbour4 = sum_neighbour4 +1
                            for k5 in range(
                                    len(adj_one[i][adj_one[i][adj_one[i][adj_one[i][adj_one[i][j][k]][k2]][k3]][k4]])):

                                node_feat_sum5 = node_feat[count + adj_one[i][
                                    adj_one[i][adj_one[i][adj_one[i][adj_one[i][j][k]][k2]][k3]][k4]][
                                    k5]] + node_feat_sum5
                                sum_neighbour5 = sum_neighbour5 + 1

                sum_neighbour = sum_neighbour + 1
           #######################该节点下的一阶、二阶、三阶、四阶、五阶邻居################
            nodefeat_2order = node_feat_sum2
            nodefeat_1order = node_feat_sum
            nodefeat_3order = node_feat_sum3
            nodefeat_4order = node_feat_sum4
            nodefeat_5order = node_feat_sum5
            ###################################################################################
            ### Generate LA

            # if i==0 and j==0:
            #     nodefeat_2order_all = node_feat_sum2
            #     nodefeat_1order_all = node_feat_sum
            #     nodefeat_3order_all = node_feat_sum3
            #     nodefeat_4order_all = node_feat_sum4
            #     nodefeat_5order_all = node_feat_sum5
            #     nodefeat_2order_all = nodefeat_2order_all.reshape((1,3))
            #     nodefeat_1order_all = nodefeat_1order_all.reshape((1,3))
            #     nodefeat_3order_all = nodefeat_3order_all.reshape((1,3))
            #     nodefeat_4order_all = nodefeat_4order_all.reshape((1,3))
            #     nodefeat_5order_all = nodefeat_5order_all.reshape((1,3))
            # else:
            #     nodefeat_2order_all =  np.row_stack((nodefeat_2order_all,node_feat_sum2))
            #     nodefeat_1order_all =np.row_stack((nodefeat_1order_all,node_feat_sum))
            #     nodefeat_3order_all =np.row_stack((nodefeat_3order_all,node_feat_sum3))
            #     nodefeat_4order_all =np.row_stack((nodefeat_4order_all,node_feat_sum4))
            #     nodefeat_5order_all = np.row_stack((nodefeat_5order_all,node_feat_sum5))
            ###########################################################################################

            # print("q+j",q+j)
            # print("nodefeat_1order_all.shape", nodefeat_1order_all)
            # print("nodefeat_1order_all.shape",nodefeat_1order_all.shape)
            # print("nodefeat_2order_all.shape", nodefeat_2order_all.shape)

#########################################################
            # nodefeat_2order = 4 * node_feat_sum2 + 5 * node_feat_sum
            # nodefeat_1order = 5 * node_feat_sum
            # nodefeat_3order = 3 * node_feat_sum3 + 4 * node_feat_sum2 + 5 * node_feat_sum
            # nodefeat_4order = 2 * node_feat_sum4 + 3 * node_feat_sum3 + 4 * node_feat_sum2 + 5 * node_feat_sum
            # nodefeat_5order = 1 * node_feat_sum5 + 2 * node_feat_sum4 + 3 * node_feat_sum3 + 4 * node_feat_sum2 + 5 * node_feat_sum

            node_feat_new0[q + j] = node_feat[count + j]
            ##########################################################
            # nodefeat_1order = 90 * (nodefeat_1order + nodefeat_2order)
            # nodefeat_2order = 90 * (nodefeat_3order + nodefeat_4order + nodefeat_5order)
            #########################################################
            nodefeat_1order = 90 * (nodefeat_3order )
            nodefeat_2order = 90 * (nodefeat_1order + nodefeat_2order + nodefeat_3order)
            node_feat_new1[q + j] = np.append(node_feat_new0[q + j], nodefeat_1order)
            # node_feat_newC[q + j] = np.append(node_feat_new1[q + j], node_feat_sum6)
            # node_feat_new2[q + j] = np.append(node_feat_new1[q + j], nodefeat_2order)
            node_feat_new2[q + j] = np.append(node_feat_new1[q + j], nodefeat_2order)

        count = count + len(adj_one[i])
        q = q + len(adj_one[i])


    q350 = 0
    count350 = 0
    # for i in range(1):
    for i in range(len(adj_one_test)):
        # print("test i", i)
        node_feat_subSUM = np.zeros((1, 3))
        for j in range(len(adj_one_test[i])):
            node_feat_subSUM = node_feat[count + count350 + j] + node_feat_subSUM


            node_feat_sum = np.zeros(3)
            node_feat_sum2 = np.zeros(3)

            node_feat_sum3 = np.zeros(3)
            node_feat_sum4 = np.zeros(3)
            node_feat_sum5 = np.zeros(3)

            sum_neighbour = 0
            sum_neighbour2 = 0
            sum_neighbour3 = 0
            sum_neighbour4 = 0
            sum_neighbour5 = 0


            for k in range(len(adj_one_test[i][j])):

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
                        for k4 in range(
                                len(adj_one_test[i][adj_one_test[i][adj_one_test[i][adj_one_test[i][j][k]][k2]][k3]])):

                            node_feat_sum4 = node_feat[
                                                 count + count350 + adj_one_test[i][
                                                     adj_one_test[i][adj_one_test[i][adj_one_test[i][j][k]][k2]][k3]][
                                                     k4]] + node_feat_sum4
                            sum_neighbour4 = sum_neighbour4 + 1
                            for k5 in range(
                                    len(adj_one_test[i][adj_one_test[i][
                                        adj_one_test[i][adj_one_test[i][adj_one_test[i][j][k]][k2]][k3]][k4]])):

                                node_feat_sum5 = node_feat[count + count350 + adj_one_test[i][
                                    adj_one_test[i][adj_one_test[i][adj_one_test[i][adj_one_test[i][j][k]][k2]][k3]][
                                        k4]][
                                    k5]] + node_feat_sum5
                                sum_neighbour5 = sum_neighbour5 +1

                sum_neighbour = sum_neighbour + 1

            nodefeat_2order = node_feat_sum2
            nodefeat_1order = node_feat_sum
            nodefeat_3order = node_feat_sum3
            nodefeat_4order = node_feat_sum4
            nodefeat_5order = node_feat_sum5
            #############################################
            nodefeat_2order_all = np.row_stack((nodefeat_2order_all, node_feat_sum2))
            nodefeat_1order_all = np.row_stack((nodefeat_1order_all, node_feat_sum))
            nodefeat_3order_all = np.row_stack((nodefeat_3order_all, node_feat_sum3))
            nodefeat_4order_all = np.row_stack((nodefeat_4order_all, node_feat_sum4))
            nodefeat_5order_all = np.row_stack((nodefeat_5order_all, node_feat_sum5))
#######################################################################
            # nodefeat_2order = 4*node_feat_sum2 + 5*node_feat_sum
            # nodefeat_1order = 5*node_feat_sum
            # nodefeat_3order = 3*node_feat_sum3 + 4*node_feat_sum2 + 5*node_feat_sum
            # nodefeat_4order = 2*node_feat_sum4 + 3*node_feat_sum3 + 4*node_feat_sum2 + 5*node_feat_sum
            # nodefeat_5order = 1*node_feat_sum5 + 2*node_feat_sum4 + 3*node_feat_sum3 + 4*node_feat_sum2 + 5*node_feat_sum

            node_feat_new0[count + q350 + j] = node_feat[count + count350 + j]
            nodefeat_1order = 90 * (nodefeat_1order + nodefeat_2order)
            nodefeat_2order = 90 * (nodefeat_3order + nodefeat_4order + nodefeat_5order)
#################################################################################
            # nodefeat_1order = 90 * (nodefeat_3order )
            # nodefeat_2order = 90 * (nodefeat_1order + nodefeat_2order+nodefeat_3order )

            node_feat_new1[count + q350 + j] = np.append(node_feat_new0[count + q350 + j], nodefeat_1order)

            node_feat_new2[count + q350 + j] = np.append(node_feat_new1[count + q350 + j], nodefeat_2order)


        count350 = count350 + len(adj_one_test[i])
        q350 = q350 + len(adj_one_test[i])
    # return node_feat_new2
    #####################################################################################
    # import pickle
    # print("nodefeat_2order_all",nodefeat_2order_all.shape)
    # fw2 = open('2order_LA.pkl', 'wb')
    # pickle.dump(nodefeat_2order_all, fw2)
    # fw1 = open('1order_LA.pkl', 'wb')
    # pickle.dump(nodefeat_1order_all, fw1)
    # fw3 = open('3order_LA.pkl', 'wb')
    # pickle.dump(nodefeat_3order_all, fw3)
    # fw4 = open('4order_LA.pkl', 'wb')
    # pickle.dump(nodefeat_4order_all, fw4)
    # fw5 = open('5order_LA.pkl', 'wb')
    # pickle.dump(nodefeat_5order_all, fw5)
    return node_feat_new1