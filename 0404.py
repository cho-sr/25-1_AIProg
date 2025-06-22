import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np



class House_Price(Dataset): #데이터 셋정의
    def __init__(self, data):
        self.features = torch.tensor(data.iloc[:, 1:-1].values, dtype=torch.float32)#텐셔로 변환
        self.targets = torch.tensor(data.iloc[:, -1].values, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


# Read the data
data = pd.read_csv('house_price_norm.csv')#파일 읽어오기

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)#훈련 테스트 분할

train_dataset = House_Price(train_data)
test_dataset = House_Price(test_data)

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)#로더 정의 셔플은 True로한다.
test_dataloader = DataLoader(test_dataset, batch_size=1)


# 2. Model training
class LinearRegression(nn.Module):#간단한 선형 모델 nn.Module 상속받아 정의
    def __init__(self, input_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)


model = LinearRegression(input_size=6)#모델 생성
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


train_losses = []
train_correlations = []

for epoch in range(100):
    epoch_loss = 0
    predictions = []
    actuals = []

    for inputs, targets in train_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets.squeeze())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        # Store predictions and actuals for correlation calculation
        predictions.extend(outputs.detach().squeeze().view(-1).tolist())#상관계수 계산을 위해 예측값들을 뽑아서 저장
        actuals.extend(targets.tolist())#상관계수를 위해 저장

    # Calculate the Pearson correlation coefficient
    correlation = np.corrcoef(actuals, predictions)[0, 1]#상관 계수 저장후 성능 지표로 활용
    train_correlations.append(correlation)
    train_losses.append(epoch_loss)



model.eval()
test_losses = []
predictions = []
targets = []

with torch.no_grad(): # 인퍼런스 진행
    for inputs, target in test_dataloader:
        output = model(inputs)
        loss = criterion(output.squeeze(), target.squeeze())
        test_losses.append(loss.item())
        predictions.append(output.item())
        targets.append(target.item())

correlation = np.corrcoef(targets, predictions)[0, 1]

