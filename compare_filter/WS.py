import numpy as np
import pywt
import torch
import time
import matplotlib.pyplot as plt
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

def wavelet_transform(signal, wavelet):
    """
    小波变换函数
    """
    coeffs = pywt.wavedec(signal, wavelet)
    return coeffs

def inverse_wavelet_transform(coeffs, wavelet):
    """
    小波反变换函数
    """
    reconstructed_signal = pywt.waverec(coeffs, wavelet)
    return reconstructed_signal

def nonlinear_soft_threshold(coeffs, threshold):
    """
    非线性软阈值函数
    """
    return np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0)

def nonlinear_wavelet_denoising(signal, wavelet, threshold):
    """
    非线性小波变换阈值法去噪函数
    """
    coeffs = wavelet_transform(signal, wavelet)
    denoised_coeffs = []
    for coeff in coeffs:
        denoised_coeff = nonlinear_soft_threshold(coeff, threshold)
        denoised_coeffs.append(denoised_coeff)
    denoised_signal = inverse_wavelet_transform(denoised_coeffs, wavelet)
    return denoised_signal
#####################################在这里设置不同的小波和阈值#####################################
'''阈值过低会导致去噪效果不佳，甚至会误判脑电信号的一些细节为噪声而被删除。阈值过高则会导致去噪效果过于明显，使得脑电信号失去原有的特征和信息。'''
# 定义小波变换参数
wavelet = 'sym8'
# 定义阈值
threshold = 0.1

states, observations = generate_signal()

denoised_signal = nonlinear_wavelet_denoising(observations, wavelet, threshold)


def loss(states, denoised_signal):
    states_tensor = torch.from_numpy(states)
    denoised_signal_tensor = torch.from_numpy(denoised_signal)
    loss_fn = torch.nn.MSELoss(reduction='mean')
    loss = loss_fn(denoised_signal_tensor, states_tensor)
    mean_loss_db = 10 * torch.log10(loss)

    ssd = mad = cosineSim = test_cc = 0
    for i in range(10):
        index = 512 * i
        ssd_index = SSD(states[index:index + 512], denoised_signal[index:index + 512])
        ssd = ssd + ssd_index
        mad = mad + MAD(states[index:index + 512], denoised_signal[index:index + 512])
        cosineSim = cosineSim + CosineSim(states[index:index + 512], denoised_signal[index:index + 512])
        test_cc = test_cc + np.corrcoef(states[index:index + 512], denoised_signal[index:index + 512])[0, 1]
    print('损失是', loss)
    print('mse是', mean_loss_db, 'db')
    print('SSD是', ssd / 10, 'uv')
    print('MAD是', mad / 10, 'uv')
    print('Cosine Sim是', cosineSim / 10)
    print('测试集的CC是', test_cc / 10)
    return loss.item()

def SSD(states, denoised_signal):
    ssd = 0
    for i in range(states.size):
        ssd = ssd + (states[i] - denoised_signal[i]) ** 2
    return ssd

def MAD(states, denoised_signal):
    mad = abs(states[0] - denoised_signal[0])
    for i in range(states.size):
        temp =  abs(states[i] - denoised_signal[i])
        mad = max(mad,temp)
    return mad

def PRD(states,ssd):
    mean = sum(states)/len(states)
    cleanssd = 0
    for i in range(states.size):
        cleanssd = cleanssd + (states[i] - mean) * (states[i] - mean)
    prd = (ssd/cleanssd) ** (1/2)
    return prd
def CosineSim(states, denoised_signal):
    cosineSim = states.dot(denoised_signal) / (np.linalg.norm(states) * np.linalg.norm(denoised_signal))
    return cosineSim

loss(states,denoised_signal)

def plot_result(states, denoised_signal):
    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(10, 6))
    ax[0].plot(states[:512])
    ax[0].set_ylabel('states')
    ax[1].plot(denoised_signal[:512])
    ax[1].set_ylabel('denoised_signal')
    plt.title('waveletshrinkage')

# 调用绘图函数
plot_result(states, denoised_signal)


end_time = time.time()
run_time = end_time - start_time
print("程序运行时间为：", run_time, "秒")

np.save('../result/hh1/WS_denoise.npy', denoised_signal)