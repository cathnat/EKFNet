"""# **Class: KalmanNet**"""
import numpy as np
import torch.nn
import torch.nn as nn
import torch.nn.functional as func
from scipy.special import factorial
from torch.nn.functional import pad
from Model.Feature_extraction import SPPCSPC
from Model.channelAttention import ChannelAttention
from Model.spatialAttention import spatial_attention
import tvm
from tvm import relay
from tvm.contrib import graph_executor


class KtModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_FC1 = 2
        self.output_FC1 = 24
        self.FC1 = nn.Sequential(
            nn.Linear(self.input_FC1, self.output_FC1),
            nn.ReLU())
        self.yolov7 = SPPCSPC()
        self.lstm = nn.LSTM(input_size=8, hidden_size=4, batch_first=True)
        self.ChannelAttention1 = ChannelAttention(2)
        self.ChannelAttention2 = ChannelAttention(4)
        self.input_FC2 = 8
        self.output_FC2 = 4
        self.FC2 = nn.Sequential(
            nn.Linear(self.input_FC2, self.output_FC2),
            nn.ReLU())
        self.input_FC3 = 16
        self.output_FC3 = 4
        self.FC3 = nn.Sequential(
            nn.Linear(self.input_FC3, self.output_FC3),
            nn.ReLU())
        self.input_FC4 = 8
        self.output_FC4 = 4
        self.FC4 = nn.Sequential(
            nn.Linear(self.input_FC4, self.output_FC4),
            nn.ReLU())
        self.input_FC5 = 8
        self.output_FC5 = 4
        self.FC5 = nn.Sequential(
            nn.Linear(self.input_FC5, self.output_FC5),
            nn.ReLU())

    def init_hidden(self):
            self.prior_Q = torch.tensor([[[0.0, 0.0, 0.0, 0.0]]])
            self.prior_R = torch.tensor([[[0.0, 0.0, 0.0, 0.0]]])
            self.prior_P = torch.tensor([[[0.0, 0.0, 0.0, 0.0]]])

    def forward(self, x, isinit=None):
        if isinit == True or isinit==None:
            self.init_hidden()
        x = x.view(1, 1, -1)
        output_FC1 = self.FC1(x)
        in_yolov7 = torch.cat((self.prior_Q, output_FC1, self.prior_P), 2)
        self.prior_P = self.yolov7(in_yolov7)
        input_FC2 = torch.cat((self.prior_P, self.prior_R), 2)
        S, _ = self.lstm(input_FC2)
        S = self.ChannelAttention1(S)
        input_FC3 = torch.cat((self.prior_P, S), 2)
        Kt = self.ChannelAttention2(input_FC3)
        Kt = self.FC3(Kt)
        input_FC4 = torch.cat((S, Kt), 2)
        output_FC4 = self.FC4(input_FC4)
        input_FC5 = torch.cat((self.prior_P, output_FC4), 2)
        self.prior_P = self.FC5(input_FC5)
        return Kt


class KalmanNetNN(torch.nn.Module):

    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        super().__init__()
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.isRelay = False

    # 泰勒展开
    def TaylorSerios(self, data):
        temp = torch.zeros(1, 2, 512)
        temp[0, 0, :] = temp[0, 1, :] = data[0, :]
        data = temp
        # 构建泰勒展开式
        basis_functions = np.array([[(1 / 512) ** k / factorial(k)] for k in range(1, 4)])
        # factorial_functions = basis_functions = torch.from_numpy(basis_functions).float().to(self.device,non_blocking = True)
        factorial_functions = basis_functions = torch.from_numpy(basis_functions).float()
        basis_functions = basis_functions.reshape((1, 1, -1))
        basis_functions = basis_functions.repeat((3, 1, 1))
        derivative_coefficients = torch.zeros(512, 3, 2)
        padded_data = pad(data, (1, 2), mode='replicate')
        weights = torch.tensor([[[1]], [[1]], [[1]]])
        for t in range(512):
            current_state = padded_data[:, :, t:t + 3]
            observations = padded_data[:, :, t + 1:t + 4]
            target_tensor = (observations - current_state).reshape(3, 1, -1)
            covariance = torch.bmm(basis_functions.permute(0, 2, 1), basis_functions)
            cross_correlation = torch.bmm(basis_functions.permute(0, 2, 1), target_tensor)
            weighted_covariance = (weights * covariance).sum(0)
            weighted_cross_correlation = (weights * cross_correlation).sum(0)
            derivatives_t = torch.matmul(torch.linalg.pinv(weighted_covariance), weighted_cross_correlation)
            derivative_coefficients[t] = derivatives_t
        self.derivative_coefficients = derivative_coefficients
        self.factorial_functions = factorial_functions

    #############
    ### Build ###
    #############
    # 构建模型
    def Build(self, ssModel):
        # 构建基础动力模型，即构建状态转移矩阵F和观测矩阵H
        self.InitSystemDynamics(ssModel.F, ssModel.H)

        self.InitKGainNet = KtModule()

    def BuildRelay(self):
        if self.isRelay == True:
            if not hasattr(KalmanNetNN, 'self.GraphModule'):
                t = torch.rand(2)
                scripted_model = torch.jit.trace(self.InitKGainNet, t)
                input_name = "input"
                shape_list = [(input_name, t.shape)]
                mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
                target = tvm.target.Target("llvm", host="llvm")
                with tvm.transform.PassContext(opt_level=3):
                    lib = relay.build(mod, target, params=params)
                dev = tvm.cpu(0)
                self.GraphModule = graph_executor.GraphModule(lib["default"](dev))
        else:
            if not hasattr(KalmanNetNN, 'self.KGN'):
                self.KGN = self.InitKGainNet

    ################################
    # Initialize System Dynamics ###
    ################################
    def InitSystemDynamics(self, F, H):
        # Set State Evolution Matrix
        self.F = F.to(self.device, non_blocking=True)
        self.F_T = torch.transpose(F, 0, 1)
        self.m = self.F.size()[0]

        # Set Observation Matrix
        self.H = H.to(self.device, non_blocking=True)
        self.H_T = torch.transpose(H, 0, 1)
        self.n = self.H.size()[0]

    ###########################
    ### Initialize Sequence ###
    ###########################
    def InitSequence(self, M1_0, M2_0):

        self.m1x_prior = M1_0.to(self.device, non_blocking=True)

        self.m1y = M2_0.to(self.device, non_blocking=True)

        self.m1x_posterior = M1_0.to(self.device, non_blocking=True)

        self.state_process_posterior_0 = M1_0.to(self.device, non_blocking=True)

    ######################
    ### Compute Priors ###
    ######################
    # 计算先验
    def step_prior(self):

        # Predict the 1-st moment of x
        # 计算xt的先验估计：X_t = F(X_t-1)
        self.m1x_prev_prior = self.m1x_prior
        # self.m1x_prior = torch.matmul(self.F, self.m1x_posterior)
        self.m1x_prior = torch.matmul(self.F, self.m1x_posterior) + torch.mm(self.derivative_coefficients[self.count].T,
                                                                             self.factorial_functions)

        # Predict the 1-st moment of y
        # 计算yt的先验估计：Y_t = H(Y_t)
        self.m1y_prev = self.m1y
        self.m1y = torch.matmul(self.H, self.m1x_prior)

    ##############################
    ### Kalman Gain Estimation ###
    ##############################
    # 计算卡尔曼增益
    def step_KGain_est(self, y):
        # 计算输入特征2：Feature 2 = yt - y_t+1|t
        Feature2 = y - torch.squeeze(self.m1y)
        Feature2_reshape = torch.squeeze(Feature2)
        Feature2_norm = func.normalize(Feature2_reshape, p=2, dim=0, eps=1e-12, out=None)
        # Kalman Gain Network Step
        # 计算卡尔曼增益
        input_name = "input"
        if self.isRelay == True:
            self.GraphModule.set_input(input_name, tvm.nd.array(Feature2_norm))
            self.GraphModule.run()
            tvm_output = self.GraphModule.get_output(0)
            KG = torch.from_numpy(tvm_output.asnumpy())
        else:
            KG = self.KGN(Feature2_norm, self.isinit)
            # Reshape Kalman Gain to a Matrix
        self.KGain = KG.reshape((self.m, self.n))

    #######################
    ### Kalman Net Step ###
    #######################
    # KalmanNet的计算总流程
    def KNet_step(self, y):
        # Compute Priors
        # 计算先验
        self.step_prior()

        # Compute Kalman Gain
        # 计算卡尔曼增益
        self.step_KGain_est(y)

        # Innovation
        # 更新项的计算
        y_obs = torch.unsqueeze(y, 1)
        dy = y_obs - self.m1y

        # Compute the 1-st posterior moment
        # 计算最优估计
        INOV = torch.matmul(self.KGain, dy)
        self.m1x_posterior = self.m1x_prior + INOV

        # return
        return torch.squeeze(self.m1x_posterior)

    ########################
    ### Kalman Gain Step ###
    ########################

    ###############
    ### Forward ###
    ###############
    def forward(self, yt, isinit):
        self.isinit = isinit
        yt = yt.to(self.device, non_blocking=True)
        return self.KNet_step(yt)
