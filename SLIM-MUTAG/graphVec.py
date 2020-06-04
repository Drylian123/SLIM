#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/5/2 23:32
# @Author : Avigdor
# @Site : 
# @File : graphVec.py
# @Software: PyCharm
import numpy as np
import torch

def graphVec(node_feat,adj_one):

    ##If you want to aggregate neighbors, you can use the code of this file
    count = 0
    node_feat = np.array(node_feat)
    if node_feat.shape[0]>350:
       node_feat_new0 = np.zeros((3371, 7))
       node_feat_new1 = np.zeros((3371, 14))
    else:
       node_feat_new0 = np.zeros((350, 7))
       node_feat_new1 = np.zeros((350, 14))

    q = 0
    for i in range(len(adj_one)):
        node_feat_subSUM = np.zeros((1, 7))
        for j in range(len(adj_one[i])):
            node_feat_subSUM = node_feat[count + j] + node_feat_subSUM
            node_feat_sum = np.zeros(7)
            node_feat_sum2 = np.zeros(7)
            node_feat_sum3 = np.zeros(7)
            sum_neighbour = 0
            sum_neighbour2 = 0
            sum_neighbour3 = 0
            # hop1
            for k in range(len(adj_one[i][j])):
                node_feat_sum = node_feat[count + adj_one[i][j][k]] + node_feat_sum
                # hop2
                for k2 in range(len(adj_one[i][adj_one[i][j][k]])):
                    node_feat_sum2 = node_feat[count + adj_one[i][adj_one[i][j][k]][k2]] + node_feat_sum2
                    sum_neighbour2 = sum_neighbour2 + 1
                    # hop3
                    for k3 in range(len(adj_one[i][adj_one[i][adj_one[i][j][k]][k2]])):
                        node_feat_sum3 = node_feat[
                                             count + adj_one[i][adj_one[i][adj_one[i][j][k]][k2]][k3]] + node_feat_sum3
                        sum_neighbour3 = sum_neighbour3 + 1
                sum_neighbour = sum_neighbour + 1
            nodefeat_3order = node_feat_sum3
            node_feat_new0[q + j] = node_feat[count + j]
            nodefeat_1order = 90 * (nodefeat_3order)
            node_feat_new1[q + j] = np.append(node_feat_new0[q + j], nodefeat_1order)
        count = count + len(adj_one[i])
        q = q + len(adj_one[i])
    node_feat = node_feat_new1.astype(np.float32)
    node_feat = torch.from_numpy(node_feat)
    node_feat = torch.relu(node_feat)
    return  node_feat
