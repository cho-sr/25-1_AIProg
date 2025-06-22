import torch
import torch.nn as nn
from torch import optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'
X = torch.FloatTensor([[0, 0], [0,1], [1,0],[1,1]]).to(device)
Y = torch.FloatTensor([[0], [1], [1],[1]]).to(device)

liner = nn.Linear(2,1)
sigmoid = nn.Sigmoid()
model = nn.Sequential(liner,sigmoid).to(device)

criterion = nn.BCELoss().to(device)
optimizer = optim.SGD(model.parameters(), lr = 1)

for epoch in range(1000):
    optimizer.zero_grad()
    hypothesis = model(X)

    cost = criterion(hypothesis, Y)
    cost.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print('Epoch:','%04d' % (epoch), 'cost:', cost.item())
with torch.no_grad():
    hypothesis = model(X)
    predicted = (hypothesis > 0.5).float()
    accuracy = (predicted == Y).float().mean()
    print('모델의 출력값(Hypothesis): ', hypothesis.detach().cpu().numpy())
    print('모델의 예측값(Predicted): ', predicted.detach().cpu().numpy())
    print('실제값(Y): ', Y.cpu().numpy())
    print('정확도(Accuracy): ', accuracy.item())