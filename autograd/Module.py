import inspect
from autograd.Tensor import Tensor
from autograd.Parameter import Parameter
from typing import Iterator


class Module:
    def parameters(self) -> Iterator[Parameter]:
        for name, value in inspect.getmembers(self):
            if isinstance(value, Parameter):
                yield value
            elif isinstance(value, Module):
                yield from value.parameters()

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()

    def forward(self, *input, **kwargs):
        raise NotImplemented("Undefined forward method")

    def __call__(self, *input, **kwargs) -> Tensor:
        return self.forward(*input, **kwargs)