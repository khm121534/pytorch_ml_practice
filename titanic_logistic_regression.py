import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd

from Own_ML_Class import LogisticRegression
import numpy as np

import seaborn as sns
data = sns.load_dataset("titanic")

# Data 전처리
# 중복 컬럼, 누락된 데이터가 많은 컬럼 제거
data.drop(['age','deck', 'class', 'embark_town', 'alive'], axis=1, inplace=True)
# 누락 데이터가 적은 컬럼은 값 채우기
# embarked의 누락된 행 번호 확인
# print(data[data['embarked'].isnull()]) # 61, 829
# 제일 빈도가 높은 데이터인 S로 replace
data.loc[[61, 829], 'embarked'] = 'S'
# object => int로
s = set(list(data['sex']))
w = set(list(data['who']))
data['class_sex'] = 0
data['class_who'] = 0
for idx, dt in enumerate(data['sex']):
    if dt == 'female':
        data.loc[idx, 'class_sex'] = 0
    else:
        data.loc[idx, 'class_sex'] = 1
for idx, dt in enumerate(data['who']):
    if dt == 'man' :
        data.loc[idx, 'class_who'] = 0
    elif dt == 'woman':
        data.loc[idx, 'class_who'] = 1
    else:
        data.loc[idx, 'class_who'] = 2
data.drop(['sex', 'who'], axis=1, inplace = True)
# 범주형 데이터 => one-hot encoding
onehot_embarked = pd.get_dummies(data['embarked'], prefix='town')
data.drop(['embarked'], axis=1, inplace=True)
data_concat = pd.concat([data, onehot_embarked], axis = 1)

x_data = data_concat.loc[:, 'pclass':'town_S']
y_data = data_concat['survived']

# float tensor로 바꾸기 위한 실수화
x_data = x_data.astype(np.float32)

from sklearn.model_selection import train_test_split
x_train_, x_test_, y_train_, y_test_ = (train_test_split(x_data, y_data, test_size = 0.2, random_state=10))

x_train = torch.tensor(x_train_.values, dtype=torch.float32)
y_train = torch.tensor(y_train_.values, dtype=torch.float32).view(-1, 1)

#print(x_train.shape) # 712, 11
# input [712, 11] output[712, 1]
model = LogisticRegression(11, 1)

optimizer = optim.SGD(model.parameters(), lr= 1e-3)
nb_epochs = 4000
for epoch in range(nb_epochs + 1):
    hx = model(x_train)
    cost = F.binary_cross_entropy(hx, y_train)

    optimizer.zero_grad()   # 미분 시 누적 방지
    cost.backward()         # cost function 미분
    optimizer.step()        # weight, bias update

    if epoch % 400 == 0:
        print(f'{epoch}/{nb_epochs} cost : {cost.item():.4f}')
print()

x_test = torch.tensor(x_test_.values, dtype=torch.float32)
y_test = torch.tensor(y_test_.values, dtype=torch.float32).view(-1, 1)
hx = model(x_test)
pred = hx >= torch.FloatTensor([0.5])
correct = 0
for i in range(len(pred)):
    if pred[i] == y_test[i]:
        correct += 1

accuracy = correct / len(pred) * 100
print(f'{correct}/{len(pred)} -> {accuracy:3.2f}%')

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
print(f"TP:{tp}, FN:{fn}, FP:{fp}, TN:{tn}")