import torch

class SystemModel:
    # F:训练样本数   H:描述状态和测量关系的矩阵，这里用的是单位矩阵   T:线性情况下的训练数据长度   T_test:线性情况下的测试数据长度
    def __init__(self, F, q, H, r, T, T_test, prior_Q=None, prior_Sigma=None, prior_S=None):
        ####################
        ### Motion Model ###
        ####################       
        self.F = F
        self.m = self.F.size()[0]

        self.q = q
        # Q:状态方程的噪声信号的协方差
        self.Q = q * q * torch.eye(self.m)
        #########################
        ### Observation Model ###
        #########################
        self.H = H
        self.n = self.H.size()[0]

        self.r = r
        # R:测量方程的噪声信号的协方差
        self.R = r * r * torch.eye(self.n)
        #Assign T and T_test
        self.T = T
        self.T_test = T_test

        if prior_Q is None:
            self.prior_Q = torch.eye(self.m)
        else:
            self.prior_Q = prior_Q

        if prior_Sigma is None:
            self.prior_Sigma = torch.zeros((self.m, self.m))
        else:
            self.prior_Sigma = prior_Sigma

        if prior_S is None:
            self.prior_S = torch.eye(self.n)
        else:
            self.prior_S = prior_S

    #####################
    ### Init Sequence ###
    #####################
    def InitSequence(self, m1x_0, m2x_0):

        self.m1x_0 = m1x_0
        self.m2x_0 = m2x_0
