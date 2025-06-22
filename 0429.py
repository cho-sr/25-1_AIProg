import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)
model = nn.Sequential(
nn.Linear(2, 1),#learnanble
nn.Sigmoid()#non learnable
)
optimizer = optim.SGD(model.parameters(), lr=3)
nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    hypothesis = model(x_train)
    cost = F.binary_cross_entropy(hypothesis, y_train)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    if epoch % 10 == 0:
        prediction = hypothesis >= torch.FloatTensor([0.5]) # predict score가 0.5를 넘으면 True
        correct_prediction = prediction.float() == y_train # 실제값과 일치하는 경우만 True
        accuracy = correct_prediction.sum().item() / len(correct_prediction) # 정확도를 계산
        print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format(epoch, nb_epochs, cost.item(), accuracy * 100,))