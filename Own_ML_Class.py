import torch
import torch.nn as nn

class LogisticRegression(nn.Module) :
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.linear = nn.Linear(self.in_dim, self.out_dim)
        self.sigmoid = nn.Sigmoid()

    """
    model = LogisticRe.
    hx = model(x_train) <-- forward 라는 메서드를 호출
    """

    def forward(self, x):
        return self.sigmoid(self.linear(x))

class LinearRegression(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.linear = nn.Linear(self.in_dim, self.out_dim)

    def forward(self, x):
        return self.linear(x)

    