import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset

class MyDataset(Dataset):
    def __init__(self,x_data,y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __getitem__(self, idx):
        return self.x_data[idx],self.y_data[idx] # 꺼내요는 능력
    def __len__(self):
        return len(self.x_data) # 얼마나 꺼내야하나


x_train = torch.FloatTensor([[73,80,75],
                             [93,88,93],
                             [89,91,90],
                             [96,98,100],
                             [73,66,70]])
y_train = torch.FloatTensor([[152],[185],[180],[196],[142]])

dataset = MyDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=False)

model = nn.Linear(3,1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

print(model.weight, model.bias)

nb_epochs = 20
for epoch in range(nb_epochs + 1):
    for samples in dataloader:

        x_train,y_train = samples

        prediction = model(x_train)
        cost = F.mse_loss(prediction, y_train)

        optimizer.zero_grad() #어디든 다있는 3줄
        cost.backward() # J 계산 -> J가 있어야 Grad 계산 가능
        optimizer.step()

    # if epoch % 100 == 0:
    #     print(f'epoch: {epoch}, loss: {cost.item()}')

print(model.weight, model.bias)

# new_data = torch.FloatTensor([[4]])
# new_pred = model(new_data)
# print(new_pred.item())
