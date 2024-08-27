from math import factorial
from torch.nn.functional import pad
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import random
from DataSet.EEG_all_DataLoader import EEG_all_Dataloader
from DataSet.GetSubset import get_subset

seed_value = 5
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

start_time = time.time()

dataloader = EEG_all_Dataloader
dataloader = dataloader(512, [0], 0, 0)
prior_loader, test_loader = get_subset(dataloader, 10)
EEG_all = prior_loader.dataset.dataset[0, 10240:15360, 0].numpy()
noise_all = prior_loader.dataset.observations[0, 10240:15360, 0].numpy()

def generate_signal():
    states = EEG_all
    observations= noise_all
    return states, observations
#-----------------------------------------

def TaylorSerios(data):
        temp = torch.zeros(1, 2, 5120)
        temp[0, 0, :] = temp[0, 1, :] = torch.from_numpy(data).float()
        data = temp
        # 构建泰勒展开式
        basis_functions = np.array([[(1 / 512) ** k / factorial(k)] for k in range(1, 4)])
        factorial_functions = basis_functions = torch.from_numpy(basis_functions).float()
        basis_functions = basis_functions.reshape((1, 1, -1))
        basis_functions = basis_functions.repeat((3, 1, 1))
        derivative_coefficients = torch.zeros(5120, 3, 2)
        padded_data = pad(data, (1, 2), mode='replicate')
        weights = torch.tensor([[[1]], [[1]], [[1]]])
        for t in range(5120):
            current_state = padded_data[:, :, t:t + 3]
            observations = padded_data[:, :, t + 1:t + 4]
            target_tensor = (observations - current_state).reshape(3, 1, -1)
            covariance = torch.bmm(basis_functions.permute(0, 2, 1), basis_functions)
            cross_correlation = torch.bmm(basis_functions.permute(0, 2, 1), target_tensor)
            weighted_covariance = (weights * covariance).sum(0)
            weighted_cross_correlation = (weights * cross_correlation).sum(0)
            derivatives_t = torch.matmul(torch.linalg.pinv(weighted_covariance), weighted_cross_correlation)
            derivative_coefficients[t] = derivatives_t
        return derivative_coefficients, factorial_functions

def extended_kalman_filter_denoising(signal):
    # 定义状态向量和状态转移矩阵
    x = np.zeros((2, 1))
    F = np.array([[1, 0], [0, 1]])

    # 定义测量矩阵和测量噪声协方差矩阵
    H = np.eye(2)
    R = 0.1

    # 定义初始状态向量和状态协方差矩阵
    x[0] = signal[0]
    x[1] = signal[1]
    P = np.array([[1000, 0], [0, 1000]])

    #初始化噪声缓存
    Q = np.zeros((2, 2))

    #存储滤波后得信号
    filtered_signal = np.zeros(len(signal))
    derivative_coefficients, factorial_functions = TaylorSerios(signal)
    # 扩展卡尔曼滤波
    for i in range(len(signal)):
        # 预测状态和协方差
        x = np.dot(F, x) + np.dot(derivative_coefficients[i].T.numpy(), factorial_functions.numpy())
        P = np.dot(F, np.dot(P, F.T)) + Q

        # 计算卡尔曼增益
        K = np.dot(P, np.dot(H.T, np.linalg.inv(np.dot(H, np.dot(P, H.T)) + R)))

        # 更新状态和协方差
        y = signal[i] - np.dot(H, x)
        x = x + np.dot(K, y)
        P = np.dot((np.eye(2) - np.dot(K, H)), P)

        # 存储滤波后的信号
        filtered_signal[i] = x[0]

        # 更新噪声缓存
        Q = np.dot((np.eye(2) - np.dot(K, H)), np.dot(Q, (np.eye(2) - np.dot(K, H)).T)) + np.dot(K, np.dot(R, K.T))

    return filtered_signal

# Generate signal
states, observations = generate_signal()

filtered_signal = extended_kalman_filter_denoising(observations)
def loss(states,filtered_signal):
    states_tensor = torch.from_numpy(states)
    filtered_signal_tensor = torch.from_numpy(np.array(filtered_signal))
    loss_fn = torch.nn.MSELoss(reduction='mean')
    loss = loss_fn(filtered_signal_tensor, states_tensor)
    mean_loss_db = 10 * torch.log10(loss)

    ssd = mad = cosineSim = test_cc = 0
    for i in range(10):
        index = 512 * i
        ssd_index = SSD(states[index:index + 512], filtered_signal[index:index + 512])
        ssd = ssd + ssd_index
        mad = mad + MAD(states[index:index + 512], filtered_signal[index:index + 512])
        cosineSim = cosineSim + CosineSim(states[index:index + 512], filtered_signal[index:index + 512])
        test_cc = test_cc + np.corrcoef(states[index:index + 512], filtered_signal[index:index + 512])[0, 1]
    print('损失是', loss)
    print('mse是', mean_loss_db,'db')
    print('SSD是', ssd / 10, 'uv')
    print('MAD是', mad / 10, 'uv')
    print('Cosine Sim是', cosineSim / 10)
    print('测试集的CC是', test_cc / 10)
    return loss.item()

def SSD(states, filtered_signal):
    ssd = 0
    for i in range(states.size):
        ssd = ssd + (states[i] - filtered_signal[i]) ** 2
    return ssd

def MAD(states, filtered_signal):
    mad = abs(states[0] - filtered_signal[0])
    for i in range(states.size):
        temp =  abs(states[i] - filtered_signal[i])
        mad = max(mad,temp)
    return mad

def PRD(states,ssd):
    mean = sum(states)/len(states)
    cleanssd = 0
    for i in range(states.size):
        cleanssd = cleanssd + (states[i] - mean) * (states[i] - mean)
    prd = (ssd/cleanssd) ** (1/2)
    return prd
def CosineSim(states, filtered_signal):
    cosineSim = states.dot(filtered_signal) / (np.linalg.norm(states) * np.linalg.norm(filtered_signal))
    return cosineSim

loss(states,filtered_signal)

# 定义绘图函数
def plot_result(states, filtered_signal):
    fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(10, 6))
    ax[0].plot(states[:512])
    ax[0].set_ylabel('states')
    ax[1].plot(filtered_signal[:512])
    ax[1].set_ylabel('filtered signal')
    ax[2].plot(noise_all[:512])
    ax[2].set_ylabel('noise_all')
    plt.title('EKF original')

# 调用绘图函数
plot_result(states, filtered_signal)

end_time = time.time()
run_time = end_time - start_time
print("程序运行时间为：", run_time, "秒")

np.save('../result/hh1/EKF_denoise.npy', filtered_signal)

