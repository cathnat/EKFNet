import torch
import torch.nn as nn
import time
import numpy as np
from matplotlib import pyplot as plt

class Pipeline_KF:

    def __init__(self):
        super().__init__()

    def setssModel(self, ssModel):
        self.ssModel = ssModel

    def setModel(self, model):
        self.model = model

    def setTrainingParams(self, n_Epochs, n_Batch, learningRate, weightDecay):
        self.N_Epochs = n_Epochs  # Number of Training Epochs
        self.N_B = n_Batch  # Number of Samples in Batch
        self.learningRate = learningRate  # Learning Rate
        self.weightDecay = weightDecay  # L2 Weight Regularization - Weight Decay

        # MSE LOSS Function
        self.loss_fn = nn.MSELoss(reduction='mean')

        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay,momentum=0.9)

    def NNTrain(self, train_input, train_target, n_CV, cv_input, cv_target):
        self.N_CV = n_CV

        MSE_cv_linear_batch = torch.empty([self.N_CV])
        self.MSE_cv_linear_epoch = torch.empty([self.N_Epochs])
        self.MSE_cv_dB_epoch = torch.empty([self.N_Epochs])

        MSE_train_linear_batch = torch.empty([self.N_B])
        self.MSE_train_linear_epoch = torch.empty([self.N_Epochs])
        self.MSE_train_dB_epoch = torch.empty([self.N_Epochs])

        ##############
        ### Epochs ###
        ##############

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0
        # cv_denoise = torch.zeros([5120])
        # cv_origin = cv_target.reshape(-1)
        # train_denoise = torch.zeros([5120])
        # train_origin = train_target.reshape(-1)
        # x_out = torch.zeros([512])
        for ti in range(0, self.N_Epochs):
            start_time = time.time()
            ###############################
            ### Training Sequence Batch ###
            ###############################
            self.optimizer.zero_grad()
            # 通过计算训练集的MSE
            # Training Mode
            self.model.train()
            isinit = True
            # if ti == 1:
            #     self.model.isRelay = True
            self.model.BuildRelay()

            Batch_Optimizing_LOSS_sum = 0

            for j in range(0, self.N_B):
                y_training = train_input[j, :, :]
                self.model.TaylorSerios(y_training)
                # 初始化X0和Y0
                self.model.InitSequence(self.ssModel.m1x_0, torch.tensor([[0.], [0.]]))

                x_out_training = torch.empty(self.ssModel.m, self.ssModel.T)
                for t in range(0, self.ssModel.T):
                    self.model.count = t
                    # 计算后验估计
                    x_out_training[:, t] = self.model(y_training[:, t], isinit)
                    isinit = False
                # x_out_training = ((x_out_training + 1) / 2) * (max_val - min_val) + min_val
                # train_denoise[j*512:(j+1)*512] = 0.5 * x_out_training[0, :] + 0.5 * x_out_training[1, :]
                # Compute Training Loss
                # 计算MSE
                LOSS = self.loss_fn(x_out_training, train_target[j, :, :].repeat((2, 1)))
                if j == 0:
                    if ti == 0:
                        minMse = LOSS.item()
                    if minMse >= LOSS.item():
                        minMse = LOSS.item()
                        x_out = 0.5 * x_out_training[0, :] + 0.5 * x_out_training[1, :]
                MSE_train_linear_batch[j] = LOSS.item()

                Batch_Optimizing_LOSS_sum = Batch_Optimizing_LOSS_sum + LOSS

            # Average
            # 计算当前批次的平均MSE
            self.MSE_train_linear_epoch[ti] = torch.mean(MSE_train_linear_batch)
            self.MSE_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_train_linear_epoch[ti])

            ##################
            ### Optimizing ###
            ##################

            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            Batch_Optimizing_LOSS_mean = Batch_Optimizing_LOSS_sum / self.N_B
            # 反向更新，更新参数
            Batch_Optimizing_LOSS_mean.requires_grad_(True)
            Batch_Optimizing_LOSS_mean.backward()

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            self.optimizer.step()

            #################################
            ### Validation Sequence Batch ###
            #################################

            # Cross Validation Mode
            # 计算验证集的MSE

            # ，model.eval():保证BN层能够用全部训练数据的均值和方差，即验证集训练过程中要保证BN层的均值和方差不变
            self.model.eval()

            for j in range(0, self.N_CV):
                y_cv = cv_input[j, :, :]
                self.model.TaylorSerios(y_cv)
                # 初始化X0和Y0
                self.model.InitSequence(self.ssModel.m1x_0, torch.tensor([[0.], [0.]]))

                x_out_cv = torch.empty(self.ssModel.m, self.ssModel.T)
                for t in range(0, self.ssModel.T):
                    # 计算后验估计
                    self.model.count = t
                    x_out_cv[:, t] = self.model(y_cv[:, t], False)
                # cv_denoise[j * 512:(j + 1) * 512] = 0.5 * x_out_cv[0, :] + 0.5 * x_out_cv[1, :]
                # Compute Training Loss
                # 计算MSE
                MSE_cv_linear_batch[j] = self.loss_fn(x_out_cv, cv_target[j, :, :].repeat((2, 1))).item()

            # Average
            # 计算当前批次的平均MSE
            self.MSE_cv_linear_epoch[ti] = torch.mean(MSE_cv_linear_batch)
            self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])

            if (self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt):
                self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
                self.MSE_cv_idx_opt = ti
                # torch.save(self.model, 'Model/model.pt')
                model_dict = self.model.state_dict()
                for key in model_dict:
                    model_dict[key] = model_dict[key].numpy()
                torch.save(model_dict, 'Model/model.pt')

            ########################
            ### Training Summary ###
            ########################
            print(ti, "MSE Training :", self.MSE_train_dB_epoch[ti], "[dB]", "MSE Validation :",
                  self.MSE_cv_dB_epoch[ti],
                  "[dB]")
            if (ti > 0):
                d_train = self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
                d_cv = self.MSE_cv_dB_epoch[ti] - self.MSE_cv_dB_epoch[ti - 1]
                print("diff MSE Training :", d_train, "[dB]", "diff MSE Validation :", d_cv, "[dB]")

            print("Optimal idx:", self.MSE_cv_idx_opt, "Optimal :", self.MSE_cv_dB_opt, "[dB]")
            end_time = time.time()
            run_time = end_time - start_time
            print("第", ti, "轮的运行时间为：", run_time, "秒")

        # x = range(self.N_Epochs)
        # plt.rcParams['axes.unicode_minus'] = False
        # plt.plot(x, self.MSE_train_dB_epoch, color='orangered', marker='o', linestyle='-', label='MSE Training')
        # plt.plot(x, self.MSE_cv_dB_epoch, color='blueviolet', marker='D', linestyle='-.', label='MSE Validation')
        # plt.legend()  # 显示图例
        # plt.xlabel("Epoch")  # X轴标签
        # plt.ylabel("MSE")  # Y轴标签
        # plt.show()

    # 与NNTrain类似
    def NNTest(self, n_Test, test_input, test_target):

        self.N_T = n_Test

        self.MSE_test_linear_arr = torch.empty([self.N_T])
        test_denoise = torch.zeros([5120])
        test_origin = test_target.reshape(-1)

        test_noise = test_input.reshape(-1)

        # MSE LOSS Function
        loss_fn = nn.MSELoss(reduction='mean')
        
        state_dict = torch.load('Model/model.pt')
        for key in state_dict:
            state_dict[key] = torch.from_numpy(state_dict[key])
        self.model.load_state_dict(state_dict)

        self.model.eval()
        self.model.isRelay = True
        self.model.BuildRelay()

        torch.no_grad()
        isinit = True

        start = time.time()

        for j in range(0, self.N_T):

            y_mdl_tst = test_input[j, :, :]
            self.model.TaylorSerios(y_mdl_tst)

            self.model.InitSequence(self.ssModel.m1x_0, torch.tensor([[0.], [0.]]))

            x_out_test = torch.empty(self.ssModel.m, self.ssModel.T)

            for t in range(0, self.ssModel.T):
                self.model.count = t
                x_out_test[:, t] = self.model(y_mdl_tst[:, t], isinit)
                isinit = False

            test_denoise[j * 512:(j + 1) * 512] = 0.5 * x_out_test[0, :] + 0.5 * x_out_test[1, :]
            self.MSE_test_linear_arr[j] = loss_fn(x_out_test, test_target[j, :, :].repeat((2, 1))).item()


        end = time.time()
        t = end - start

        # Average
        self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
        self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)

        # Standard deviation
        self.MSE_test_dB_std = torch.std(self.MSE_test_linear_arr, unbiased=True)
        self.MSE_test_dB_std = 10 * torch.log10(self.MSE_test_dB_std)

        # Print MSE Cross Validation
        str = "MSE Test:"
        print(str, self.MSE_test_dB_avg, "[dB]")
        str = "STD Test:"
        print(str, self.MSE_test_dB_std, "[dB]")
        # Print Run Time
        print("Inference Time:", t)

        test_ssd = test_mad = test_prd = test_cosineSim = 0
        for i in range(10):
            index = 512 * i
            test_ssd_index = self.SSD(test_origin[index:index + 512].detach().numpy(),
                                      test_denoise[index:index + 512].detach().numpy())
            test_ssd = test_ssd + test_ssd_index
            test_mad = test_mad + self.MAD(test_origin[index:index + 512].detach().numpy(),
                                           test_denoise[index:index + 512].detach().numpy())
            test_prd = test_prd + self.PRD(test_origin[index:index + 512].detach().numpy(), test_ssd_index)
            test_cosineSim = test_cosineSim + self.CosineSim(test_origin[index:index + 512].detach().numpy(),
                                                             test_denoise[index:index + 512].detach().numpy())
        print('测试集的SSD是', test_ssd / 10, 'uv')
        print('测试集的MAD是', test_mad / 10, 'uv')
        print('测试集的PRD是', test_prd, '%')
        print('测试集的Cosine Sim是', test_cosineSim / 10)

        np.save('groundtruth.npy', test_origin.detach().numpy())
        np.save('noise.npy', test_noise.detach().numpy())
        np.save('denoise.npy', test_denoise.detach().numpy())



        # x_data = range(1, 11)
        # y_data = self.MSE_test_linear_arr
        # plt.rcParams["axes.unicode_minus"] = False
        # plt.bar(x_data, y_data)
        # plt.xlabel("data")
        # plt.ylabel("MSE")
        # plt.show()

        self.plot_result(test_origin, test_noise, test_denoise)

        return [self.MSE_test_linear_arr, self.MSE_test_linear_avg, self.MSE_test_dB_avg, x_out_test]

    def SSD(self, states, filtered_signal):
        ssd = 0
        for i in range(512):
            ssd = ssd + (states[i] - filtered_signal[i]) ** 2
        return ssd

    def MAD(self, states, filtered_signal):
        mad = abs(states[0] - filtered_signal[0])
        for i in range(512):
            temp = abs(states[i] - filtered_signal[i])
            mad = max(mad, temp)
        return mad

    def PRD(self, states, ssd):
        mean = sum(states) / len(states)
        cleanssd = 0
        for i in range(512):
            cleanssd = cleanssd + (states[i] - mean) * (states[i] - mean)
        prd = (ssd / cleanssd) ** (1 / 2)
        return prd

    def CosineSim(self, states, filtered_signal):
        cosineSim = states.dot(filtered_signal) / (np.linalg.norm(states) * np.linalg.norm(filtered_signal))
        return cosineSim

    def plot_result(self, target, noise, denoise):
        fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(10, 6))

        ax[0].plot(target[:1024].detach().numpy())
        ax[0].set_ylabel('ground truth')

        ax[1].plot(noise[:1024].detach().numpy())
        ax[1].set_ylabel('noise')

        ax[2].plot(denoise[:1024].detach().numpy())
        ax[2].set_ylabel('denoise')

        plt.title('denoiseNet')
        plt.show()

