from autograd.Tensor import Tensor
from autograd.Parameter import Parameter
from autograd.Module import Module
from autograd.Function import *
from typing import Optional, NamedTuple, Union


class Linear(Module):
    def __init__(self,
                 input_size: int = None,
                 output_size: int = None,
                 bias: bool = True) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.bias = bias
        self.w = Parameter(input_size, output_size)

        if self.bias:
            self.b = Parameter(output_size)
        else:
            self.b = None

    def forward(self, input: Tensor) -> Tensor:
        return linear(input, self.w, self.b)


class ReLU(Module):
    def __init__(self):
        super(self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return relu(input)


class Sigmoid(Module):
    def __init__(self):
        super(self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return sigmoid(input)


class Tanh(Module):
    def __init__(self):
        super(self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return tanh(input)


class Threshold(Module):
    def __init__(self,
                 thr: float = None,
                 val: float = None):
        super(self).__init__()
        self.thr = thr
        self.val = val

    def forward(self, input: Tensor) -> Tensor:
        return threshold(input, self.thr, self.val)


class Softmax(Module):
    def __init__(self):
        super(self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return softmax(input)
