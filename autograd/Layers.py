from autograd.Tensor import Tensor
from autograd.Parameter import Parameter
from autograd.Module import Module
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
        return (input @ self.w) + (self.b or 0)

