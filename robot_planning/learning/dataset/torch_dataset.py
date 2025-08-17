import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from .sampling import GMMDist


class TorchDataset(Dataset):
    """All subject dataset class.

    Parameters
    ----------
    split_ids : list
        ids list of training or validation or traning data.

    Attributes
    ----------
    split_ids

    """

    def __init__(self, n_data_points):
        super(TorchDataset, self).__init__()
        self.data = GMMDist(dim=2).sample((n_data_points,))

    def __getitem__(self, index):
        # Read only specific data and convert to torch tensors
        x = torch.from_numpy(self.features[index]).type(torch.float32)
        y = torch.from_numpy(self.labels[index, :]).type(torch.float32)
        return x

    def __len__(self):
        return self.features.shape[0]
