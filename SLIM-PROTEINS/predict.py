from __future__ import print_function

import os
import sys
import numpy as np
import torch
import random
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn

import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import pdb

sys.path.append('%s/lib' % os.path.dirname(os.path.realpath(__file__)))
from pytorch_util import weights_init

class MLPRegression(nn.Module):
    def __init__(self, input_size, hidden_size, with_dropout=False):
        super(MLPRegression, self).__init__()

        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, 1)
        self.with_dropout = with_dropout

        weights_init(self)

    def forward(self, x, y = None):
        h1 = self.h1_weights(x)
        h1 = F.relu(h1)

        if self.with_dropout:
            h1 = F.dropout(h1, training=self.training)
        pred = self.h2_weights(h1)[:, 0]

        if y is not None:
            y = Variable(y)
            mse = F.mse_loss(pred, y)
            mae = F.l1_loss(pred, y)
            mae = mae.cpu().detach()
            return pred, mae, mse
        else:
            return pred

class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_class, with_dropout=False):
        super(MLPClassifier, self).__init__()

        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h1_weightsqkq = nn.Linear(input_size, hidden_size)
        self.h1_weightsPU = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, num_class)
        self.h2_weights_bin = nn.Linear(7, 2)
        self.h2_weights_bin1 = nn.Linear(100, 50)
        self.h2_weights_bin12 = nn.Linear(50, 10)
        self.h2_weights_bin13 = nn.Linear(10, 2)

        self.h2_weights_qkq1_1000_100 = nn.Linear(1000, 100)
        self.h2_weights_qkq1_100_10 = nn.Linear(100, 10)
        ###########100x10##############
        self.h2_weights_qkq1_100_10 = nn.Linear(100, 10)
        self.h2_weights_qkq1_1000_500 = nn.Linear(1000, 500)
        self.h2_weights_qkq1_500_200 = nn.Linear(500, 200)
        self.h2_weights_qkq1_200_100 = nn.Linear(200, 100)
        self.h2_weights_qkq1_200_100_1 = nn.Linear(200, 100)
        self.h2_weights_qkq1_100_50 = nn.Linear(100, 50)
        self.h2_weights_qkq1_50_2 = nn.Linear(50, 2)

        ###########10000##############
        self.h2_weights_qkq1 = nn.Linear(10000,5000)
        self.h2_weights_qkq2 = nn.Linear(5000, 1000)
        self.h2_weights_qkq3 = nn.Linear(1000, 500)
        self.h2_weights_qkq4 = nn.Linear(500, 10)
        self.h2_weights_qkq5 = nn.Linear(10, 2)
        self.h2_weights_bin = nn.Linear(10, 2)
        self.with_dropout = with_dropout
        self.a = nn.Parameter(torch.ones(size=(1, 1)))
        self.b = nn.Parameter(torch.ones(size=(1, 1)))
        self.c = nn.Parameter(torch.ones(size=(1, 1)))
        weights_init(self)





    def forward(self,  node_feat_x,bin,qkq,q_sub,bin1 ,y=None, ):


        bin1 = bin1.tolist()
        bin1=[bin1]
        bin1=np.array(bin1)
        bin1=torch.from_numpy(bin1).type(torch.FloatTensor)
        bin1=bin1.cuda()

        bin2=bin1
        # print()
        bin1=self.h2_weights_bin1(bin1)
        bin1 = self.h2_weights_bin12(bin1)
        bin1 = self.h2_weights_bin13(bin1)
        bin1 = F.softmax(bin1, dim=1)



        qkq=qkq.cuda()




        ppt = torch.mm(bin2.t(), bin2)
        qkq = torch.div(qkq, (ppt+0.001))


        qkq = self.h2_weights_qkq1_100_10(qkq)
        qkq = qkq.reshape(1, 1000)
        qkq = self.h2_weights_qkq1_1000_500(qkq)
        qkq = self.h2_weights_qkq1_500_200(qkq)
        qkq = self.h2_weights_qkq1_200_100(qkq)

        qkq = self.h2_weights_qkq1_100_50(qkq)
        qkq = self.h2_weights_qkq1_50_2(qkq)
        bin2 = self.h2_weights_qkq1_100_50(bin2)
        bin2 = self.h2_weights_qkq1_50_2(bin2)


        a1=self.a * self.a/(self.a*self.a+self.b*self.b)
        b1=self.b * self.b/(self.a*self.a+self.b*self.b)
        qkq = a1* qkq + b1 * bin2

        if y is not None:
            y = Variable(y)
            qkq = F.log_softmax(qkq, dim=1)

            loss = F.nll_loss(qkq, y)
            pred = qkq.data.max(1, keepdim=True)[1]
            acc = pred.eq(y.data.view_as(pred)).cpu().sum().item() / float(y.size()[0])
            return qkq, loss, acc
        else:
            return qkq