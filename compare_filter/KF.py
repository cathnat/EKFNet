import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import torch
import time
import random
from DataSet.EEG_all_DataLoader import EEG_all_Dataloader
from DataSet.GetSubset import get_subset

seed_value = 42
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


class KalmanFilter:

    def __init__(self, x0, P, F, Q, H, R):
        self.x = x0
        self.P = P
        self.F = F
        self.Q = Q
        self.H = H
        self.R = R

    def predict(self):  # @ is the matrix multiplication operator in Python
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def updata(self, z):


        K = self.P @ self.H.T @ inv(self.H @ self.P @ self.H.T + self.R)

        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(len(self.x)) - K @ self.H) @ self.P



def main():
    x, y = generate_signal() # states observation

    # 初始化卡尔曼滤波器

    x0 = np.array([y[0]])       # initial value
    P = np.eye(1)*0.01          # Error covariance matrix
    F = np.array([[1]])         # State Transfer Matrix
    Q = np.array([[0.1]])       # Covariance of the random signal w
    H = np.array([[1]])         # Observation Matrix
    R = np.array([[0.1]])       # The covariance of the random signal v
    kf = KalmanFilter(x0, P, F, Q, H, R)    # 设置形参

    # 执行滤波
    y_filtered = []
    for i in range(len(y)):
        kf.predict()
        kf.updata(y[i])
        y_filtered.append(kf.x[0])

    return y_filtered

states, observations = generate_signal()
y_filtered = main()


def loss(states,y_filtered):
    states_tensor = torch.from_numpy(states)
    y_filtered_tensor = torch.from_numpy(np.array(y_filtered))
    loss_fn = torch.nn.MSELoss(reduction='mean')
    loss = loss_fn(y_filtered_tensor, states_tensor )
    mean_loss_db = 10 * torch.log10(loss)

    ssd = mad = cosineSim = test_cc = 0
    for i in range(10):
        index = 512 * i
        ssd_index = SSD(states[index:index + 512], y_filtered[index:index + 512])
        ssd = ssd + ssd_index
        mad = mad + MAD(states[index:index + 512], y_filtered[index:index + 512])
        cosineSim = cosineSim + CosineSim(states[index:index + 512], y_filtered[index:index + 512])
        test_cc = test_cc + np.corrcoef(states[index:index + 512], y_filtered[index:index + 512])[0, 1]
    print('损失是', loss)
    print('mse是', mean_loss_db, 'db')
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

loss(states,y_filtered)

def plot_result(states, y_filtered):
    fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(10, 6))
    ax[0].plot(states[:512])
    ax[0].set_ylabel('states')

    ax[1].plot(y_filtered[:512])
    ax[1].set_ylabel('y_filtered')

    ax[2].plot(noise_all[:512])
    ax[2].set_ylabel('noise_all')

    plt.title('kalman filter')

# 调用绘图函数
plot_result(states, y_filtered)

end_time = time.time()
run_time = end_time - start_time
print("程序运行时间为：", run_time, "秒")

np.save('../result/hh1/KF_denoise.npy', y_filtered)
