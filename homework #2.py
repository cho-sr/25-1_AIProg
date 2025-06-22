import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd

class WineDataset(Dataset):
    def __init__(self,csv):
        data = pd.read_csv(csv).to_numpy()

        self.x = torch.tensor(data[:, :-1])
        self.y = torch.tensor(data[:, -1])
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]



dataset = WineDataset('winequality-red-rev.csv')
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
x_batch, y_batch = next(iter(dataloader))
print(x_batch.shape)
print(y_batch.shape)