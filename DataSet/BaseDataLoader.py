import numpy as np
from colorednoise import powerlaw_psd_gaussian as ppg
from matplotlib import pyplot as plt
from torch.utils.data.dataloader import Dataset
import os
import torch
from scipy.signal import find_peaks



class BaseEEGLoader(Dataset):

    def __init__(self, datapoints: int, samples: list, snr_db: int, noise_color: int = 0):
        super(BaseEEGLoader, self).__init__()

        print('prepare EEG_signals')
        self.datapoints = datapoints

        self.file_location = os.path.dirname(os.path.realpath(__file__))

        # Load dataset
        self.dataset, self.fs ,self.min_val , self.max_val = self.load_data(samples)

        # Get dataset dimensions
        self.samples, self.signal_length, self.num_channels = self.dataset.shape

        # Add gaussian white noise
        self.observations = self.add_noise(self.dataset, snr_db, noise_color)




    def load_data(self, samples: list) -> (torch.Tensor, int):
        """
        Load the dataset as a tensor with dimensions: (Samples, Time, channel)
        :param samples: Array of samples to choose from
        :return: Raw dataset and sampling frequency
        """
        raise NotImplementedError

    def add_noise(self, dataset: torch.Tensor, snr_db: int, noise_color: int) -> torch.Tensor:
        """
        Add noise of a specified color and snr
        :param snr_db: Signal to noise ratio in decibel
        :param noise_color: Color of noise 0: white, 1: pink, 2: brown, ...
        :return: Tensor of noise data
        """
        # Calculate signal power along time domain
        signal_power_db = 10 * torch.log10(dataset.var(1) + dataset.mean(1) ** 2)

        # Calculate noise power
        noise_power_db = signal_power_db - snr_db
        noise_power = torch.pow(10, noise_power_db / 20)

        # Set for reproducibility
        random_state = 42

        # # Generate noise
        # noise = [ppg(noise_color, self.signal_length, random_state=random_state) for _ in range(self.num_channels)]
        # noise = torch.tensor(np.array(noise)).T.float() * noise_power

        EMG_noise = np.load('DataSet/EMG_all_epochs.npy').reshape(-1, 1)[:650000]
        EOG_noise = np.load('DataSet/EOG_all_epochs.npy').reshape(-1, 1)[:650000]
        EMG_min_val = np.min(EMG_noise)
        EOG_min_val = np.min(EOG_noise)
        EMG_max_val = np.max(EMG_noise)
        EOG_max_val = np.max(EOG_noise)
        EMG_noise = 4 * (EMG_noise - EMG_min_val) / (EMG_max_val - EMG_min_val) - 2
        EOG_noise = 10 * (EOG_noise - EOG_min_val) / (EOG_max_val - EOG_min_val) - 5
        EMG_noise = np.repeat(EMG_noise,2,axis=1)
        EOG_noise = np.repeat(EOG_noise,2,axis=1)
        EMG_noise = torch.tensor(EMG_noise).float() * noise_power
        EOG_noise = torch.tensor(EOG_noise).float() * noise_power
        noise = EMG_noise + EOG_noise

        # Add noise
        noisy_data = self.dataset + noise

        # target = self.dataset[0,0:1024,0]
        # addnoise = noise[0:1024,0]
        # newdata = noisy_data[0,0:1024,0]
        # fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(10, 6))
        #
        # ax[0].plot(target[:1024].detach().numpy())
        # ax[0].set_ylabel('ground truth')
        #
        # ax[1].plot(addnoise[:1024].detach().numpy())
        # ax[1].set_ylabel('noise')
        #
        # ax[2].plot(newdata[:1024].detach().numpy())
        # ax[2].set_ylabel('addnoise')
        #
        # plt.show()

        return noisy_data