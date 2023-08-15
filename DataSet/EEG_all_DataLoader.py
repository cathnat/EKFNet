import numpy as np
import torch
from DataSet.BaseDataLoader import BaseEEGLoader
import _pickle as pickle

class EEG_all_Dataloader(BaseEEGLoader):

    def __init__(self, datapoints: int,samples: list, snr_db: int, noise_color: int = 0):
        super(EEG_all_Dataloader,self).__init__(datapoints, samples, snr_db, noise_color)

    def load_data(self, samples: list) -> (torch.Tensor, int):

        # Define sampling frequency
        samples_per_second = 512

        # EEGdata
        eeg_data = np.load('DataSet/EEG_all_epochs.npy')
        eeg_data = eeg_data.reshape(-1, 1)[:650000]

        min_val = np.min(eeg_data)
        max_val = np.max(eeg_data)

        eeg_data = ((eeg_data - min_val) / (max_val - min_val)) * 2 - 1
        #eeg_data = (eeg_data - np.min(eeg_data, axis=0)) / (np.max(eeg_data, axis=0) - np.min(eeg_data, axis=0))

        dataset = torch.from_numpy(eeg_data).float()
        # dataset = dataset.unsqueeze(0)   torch.Size([1, 2311168, 64])
        dataset = dataset.unsqueeze(0).expand(-1, -1, 2)
        # dataset = dataset.unsqueeze(0)

        # Convert to torch tensor
        dataset = torch.tensor(np.array(dataset)).float()

        return dataset, samples_per_second , min_val, max_val

    def __len__(self):
        return 42

