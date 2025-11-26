import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd

from Own_ML_Class import LinearRegression
import numpy as np

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

import pickle

with open("data/housing.pkl", "rb") as f:
    house = pickle.load(f)

feature = house['data']
target = house['target']

train_len = int(len(feature) * 0.8)

# 데이터 정규화
scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_data = scaler_x.fit_transform(feature[:train_len])
y_data = scaler_y.fit_transform(target[:train_len].reshape(-1, 1)).flatten()

x_test_ = scaler_x.transform(feature[train_len:])
y_test_ = scaler_y.transform(target[train_len:].reshape(-1, 1)).flatten()

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data).view(-1, 1)
x_test = torch.FloatTensor(x_test_)
y_test = torch.FloatTensor(y_test_).view(-1, 1)

# print(x_train.shape) # 4128, 8

dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=96, shuffle=True)

model = LinearRegression(8, 1)
optimizer = optim.SGD(model.parameters(), lr = 1e-3)

# MSELoss 는 클래스이므로 먼저 객체를 생성하고 그 객체에 값을 넣어야함
getMSE = nn.MSELoss()

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    for x_batch, y_batch in dataloader:
        hx = model(x_batch)
        cost = getMSE(hx, y_batch)

        optimizer.zero_grad()  # 미분 시 누적 방지
        cost.backward()  # cost function 미분
        optimizer.step()  # weight, bias update

    if(epoch % 100 == 0) :
        print(f"{epoch}/{nb_epochs} cost : {cost.item():.4f}")

# inference
from skimage.metrics import mean_squared_error
hx = model(x_test)
# mse는 np배열이나 list로 바꿔줘야 함
hx_np = hx.detach().numpy()
y_test_np = y_test.detach().numpy()

mse = mean_squared_error(y_test_np, hx_np)
print(f'Mean Square Error : {mse:.4f}')

