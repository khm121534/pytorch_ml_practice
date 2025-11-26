import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix

from Own_ML_Class import LogisticRegression
import numpy as np


# 당뇨별 판별 데이터 Read
data = np.loadtxt('data/data-03-diabetes.csv', delimiter=',', dtype=np.float32)

# 마지막 20개(row) -> test data, 나머지 -> train data
# 마지막 컬럼 -> GT
train_len = len(data) - 20
x_data = data[:train_len, : 8]
y_data = data[:train_len, 8:]
test_x = data[train_len : , : 8]
test_y = data[train_len : , 8:]

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

# input [739, 8] * [8, 1] = output [739, 1]
model = LogisticRegression(8, 1)

optimizer = optim.SGD(model.parameters(), lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    hx = model(x_train)
    cost = F.binary_cross_entropy(hx, y_train)

    optimizer.zero_grad()   # 미분 시 누적 방지
    cost.backward()         # cost function 미분
    optimizer.step()        # weight, bias update

    if epoch % 100 == 0:
        print(f"{epoch}/{nb_epochs} cost : {cost.item():.4f}")

x_test = torch.FloatTensor(test_x)
y_test = torch.FloatTensor(test_y)

hx = model(x_test)
pred = hx >= torch.FloatTensor([0.5])
correct = 0
for i in range(len(pred)):
    if pred[i] == y_test[i]:
        correct += 1

accuracy = correct / len(pred) * 100
print(f'{correct}/{len(pred)} -> {accuracy}%')

# Confusion Matrix
from sklearn import metrics
cm = metrics.confusion_matrix(pred, y_test)
print(cm)

tp=0
fp=0
fn=0
tn=0
for i in range(len(pred)):
    if (pred[i] == True):
        if(y_test[i] == 1):
            tp += 1
        else:
            fp += 1
    else:
        if(y_test[i] == 1):
            fn += 1
        else:
            tn += 1
print(f"[TP:{tp}, FP:{fp}\n"
      f" FN:{fn}, TN:{tn}]")
