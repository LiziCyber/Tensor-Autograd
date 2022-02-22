import numpy as np
from autograd import Tensor, Module, Parameter
from autograd.Function import tanh
from autograd.Optimizer import SGD
from autograd.Layers import Linear
from typing import List

n_samples = 1000
x_train = Tensor(np.random.randn(n_samples, 3))
coef = Tensor(np.ones((3, 1)))
bias = Tensor(10)
y_train = x_train @ coef + bias


class Model(Module):
    def __init__(self, input_size, output_size) -> None:
        self.l1 = Linear(input_size, output_size)

    def forward(self, input: Tensor) -> Tensor:
        return self.l1(input)


model = Model(3, 1)
optimizer = SGD(model, lr=0.01)

for epoch in range(1000):
    epoch_loss = 0.0

    model.zero_grad()
    predicted = model.forward(x_train)
    actual = y_train
    errors = (predicted - actual)
    loss = (errors * errors).sum() / n_samples

    loss.backward()
    optimizer.step(model)

    print(epoch, loss.data)

print(model.l1.w)
print(model.l1.b)