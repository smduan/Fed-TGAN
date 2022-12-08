import torch
from torch.utils.data.dataset import Dataset

class MyTabularDataset(Dataset):

    def __init__(self, dataset):
        """
        :param dataset:
        """
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        data = self.data[index]

        return torch.tensor(data).float()



