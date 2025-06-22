import csv
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class HousePriceDataset(Dataset):
    def __init__(self, path, is_train=True):
        self.is_train = is_train
        self.data = self.csv_read(path)
        self.features = self.data[:, 1:7]
        self.targets = self.data[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(self.features,
                                                            self.targets,
                                                            test_size=0.2,
                                                            random_state=42)
        self.X_train = torch.FloatTensor(X_train)
        self.X_test = torch.FloatTensor(X_test)
        self.y_train = torch.FloatTensor(y_train).view(-1, 1)
        self.y_test = torch.FloatTensor(y_test).view(-1, 1)

    def __getitem__(self, idx):
        if self.is_train:
            return self.X_train[idx, :], self.y_train[idx]
        else:
            return self.X_test[idx, :], self.y_test[idx]

    def __len__(self):
        if self.is_train:
            return len(self.X_train)
        else:
            return len(self.X_test)

    def csv_read(self, path):
        data = []
        with open(path) as f:
            reader = csv.reader(f)
            _ = next(reader)
            for idx, row in enumerate(reader):
                data.append(row)
        return np.array(data, dtype=float)

class DataConstructor:
    def __init__(self, path):
        self.path = path
        self.dataset = HousePriceDataset(path=self.path, is_train=True)
        self.train_loader = DataLoader(self.dataset, batch_size=16, shuffle=True, drop_last=False)
        self.valid_loader = DataLoader(self.dataset, batch_size=1, shuffle=False, drop_last=False)
        self.dataset.is_train = False
        self.test_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
    def get_loaders(self):
        return self.dataset, self.train_loader, self.valid_loader, self.test_loader

class MyModel(nn.Module):
    def __init__(self, n_feature):
        super(MyModel, self).__init__()
        self.linear1 = nn.Linear(n_feature, 10)
        self.linear2 = nn.Linear(10, 10)
        self.linear3 = nn.Linear(10, 1)
        self.activation = nn.ReLU(inplace=True)
        self.apply(self.initialize_weights)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        return x

    def initialize_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)

class Trainer:
    def __init__(self, data_constructor, lr=0.05):
        self.data_constructor = data_constructor
        self.dataset = data_constructor.dataset
        self.train_loader = data_constructor.train_loader
        self.lr = lr
        self.w_num = self.dataset.X_train.shape[1]
        self.model = MyModel(self.w_num)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(list(filter(lambda x: x.requires_grad, self.model.parameters())),
                                         lr=self.lr)

    def train(self, num_epochs=100):
        self.loss_history = []
        self.accuracy_history = []
        self.num_epochs = num_epochs
        for epoch in range(self.num_epochs):
            for x_train, y_train in self.train_loader:
                self.model.train()
                validation = Tester(self.data_constructor, valid=True)
                valid_out = validation.test(self.model, self.criterion)
                r, _ = stats.pearsonr(valid_out['predictions'], valid_out['targets'])

                output = self.model(x_train)
                loss = self.criterion(output, y_train)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.loss_history.append(loss.item())
                self.accuracy_history.append(r)
            print(f'epoch {epoch+1} loss: {loss.item():.4f}, acc: {r}')
        self.out = {"losses": self.loss_history, "accuracies": self.accuracy_history}
        return self.out

class Tester:
    def __init__(self, data_constructor, valid=False):
        if valid:
            self.loader = data_constructor.valid_loader
        self.loader = data_constructor.test_loader

    def test(self, model, criterion):
        self.losses = []
        self.preds = []
        self.targets = []
        with torch.no_grad():
            model.eval()
            for x_test, y_test in self.loader:
                pred = model(x_test)
                loss = criterion(pred, y_test)
                self.losses.append(loss.item())
                self.preds.append(pred.item())
                self.targets.append(y_test.item())
        self.out = {'losses': self.losses, 'predictions': self.preds, 'targets': self.targets}
        return self.out

class Visualizer:
    def train_log_plot(self, loss_history, accuracy_history):
        self.loss_history = loss_history
        self.accuracy_history = accuracy_history
        plt.rcParams['axes.grid'] = True
        fig, axes = plt.subplots(2, 1)
        axes[0].plot(self.loss_history, 'r--')
        axes[0].set_title('loss')
        axes[1].plot(self.accuracy_history, 'b--')
        axes[1].set_title('train accuracy')
        plt.tight_layout()
        plt.show()

    def draw_jointplot(self, preds, targets):
        df = pd.DataFrame({'prediction': preds, 'target': targets})
        sns.jointplot(data=df, x='prediction', y='target', kind='reg',
                      marker='o',
                      scatter_kws=dict(s=2),
                      marginal_kws=dict(bins=25, fill=True),
                      marginal_ticks=False)
        plt.show()

    def draw_scatterplot(self, preds, targets):
        plt.plot(preds, targets,
                 linestyle='none',
                 marker='o',
                 markersize=5,
                 color='red',
                 alpha=0.5)
        plt.xlabel('prediction', fontsize=14)
        plt.ylabel('target', fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # 데이터 구축
    data_constructor = DataConstructor(path='./house_price_norm.csv')

    # 학습
    lr = 1e-3
    trainer = Trainer(data_constructor, lr=lr)
    train_out = trainer.train(num_epochs=100)

    # 테스트
    tester = Tester(data_constructor, valid=False)
    test_out = tester.test(trainer.model, trainer.criterion)

    # 가시화 및 결과확인
    visualizer = Visualizer()
    visualizer.train_log_plot(train_out['losses'], train_out['accuracies'])
    visualizer.draw_jointplot(test_out['predictions'], test_out['targets'])
    visualizer.draw_scatterplot(test_out['predictions'], test_out['targets'])
    r, _ = stats.pearsonr(test_out['predictions'], test_out['targets'])
    print(f'[*] correlation score is {r:.4f}')
    print(f'[*] MSE is {np.mean(test_out["losses"]):.4f}')