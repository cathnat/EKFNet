import numpy as np
from matplotlib import pyplot as plt


def SSD(states, filtered_signal):
    ssd = 0
    for i in range(512):
        ssd = ssd + (states[i] - filtered_signal[i]) ** 2
    return ssd


def MAD(states, filtered_signal):
    mad = abs(states[0] - filtered_signal[0])
    for i in range(512):
        temp = abs(states[i] - filtered_signal[i])
        mad = max(mad, temp)
    return mad


def PRD(states, ssd):
    mean = sum(states) / len(states)
    cleanssd = 0
    for i in range(512):
        cleanssd = cleanssd + (states[i] - mean) * (states[i] - mean)
    prd = (ssd / cleanssd) ** (1 / 2)
    return prd


def CosineSim(states, filtered_signal):
    cosineSim = states.dot(filtered_signal) / (np.linalg.norm(states) * np.linalg.norm(filtered_signal))
    return cosineSim


def plot_result(target, noise, denoise):
    fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(10, 6))

    ax[0].plot(target[:1024].cpu().detach().numpy())
    ax[0].set_ylabel('ground truth')

    ax[1].plot(noise[:1024].cpu().detach().numpy())
    ax[1].set_ylabel('noisy signal')

    ax[2].plot(denoise[:1024].cpu().detach().numpy())
    ax[2].set_ylabel('denoise signal')

    plt.show()