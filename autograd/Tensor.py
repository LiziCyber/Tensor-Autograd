import numpy as np
from typing import List, Union, Callable, NamedTuple, Optional

Arrayable = Union[float, list, np.ndarray]


class Dependency(NamedTuple):
    tensor: 'Tensor'
    grad_fn: Callable


def ensure_array(arrayable: Arrayable) -> np.ndarray:
    if isinstance(arrayable, np.ndarray):
        return arrayable
    else:
        return np.array(arrayable)


class Tensor:
    def __init__(self,
                 data: Arrayable,
                 requires_grad: bool = False,
                 depends_on: List[Dependency] = None
                 ) -> None:
        self.data = ensure_array(data)
        self.requires_grad = requires_grad
        self.depends_on = depends_on or []
        self.shape = self.data.shape
        self.grad: Optional['Tensor'] = None

        if self.requires_grad:
            self.zero_grad()

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def zero_grad(self):
        self.grad = Tensor(np.zeros_like(self.data))

    def backward(self, grad: 'Tensor' = None):
        assert self.requires_grad, "called backward on non-requires-grad tensor"

        if grad is None:
            grad = Tensor(np.ones_like(self.data))

        self.grad.data = self.grad.data + grad.data

        for dependency in self.depends_on:
            backward_grad = dependency.grad_fn(grad.data)
            dependency.tensor.backward(Tensor(backward_grad))

    def sum(self):
        return tensor_sum(self)

    def __add__(self, other: 'Tensor') -> 'Tensor':
        return tensor_add(self, other)

    def __mul__(self, other):
        return tensor_mul(self, other)


def tensor_sum(t: Tensor) -> Tensor:
    """
    Return a 0-tensor which is the sum over all elements of the input Tensor.
    :param t: input tensor
    :return: a 0-Tensor
    """
    # TBD sum over selected axes
    data = t.data.sum()
    requires_grad = t.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * np.ones_like(t.data)

        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)


def tensor_add(a: Tensor, b: Tensor) -> Tensor:
    """
    Return the sum of two Tensor. Broadcasting is allowed.
    :param a: left tensor
    :param b: right tensor
    :return: a tensor representing the sum of the two tensor
    """
    data = a.data + b.data
    requires_grad = a.requires_grad or b.requires_grad
    depends_on: List[Dependency] = []

    if a.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            # added ndim explicitly
            ndims_added = grad.ndim - a.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            # no added ndim but still broadcasting through the dims which have size 1
            for i, size in enumerate(a.shape):
                if size == 1:
                    grad = grad.sum(axis=i, keepdims=True)
                else:
                    break

            return grad

        depends_on.append(Dependency(a, grad_fn1))

    if b.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            # added ndim explicitly
            ndims_added = grad.ndim - b.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            # no added ndim but still broadcasting through the dims which have size 1
            for i, size in enumerate(b.shape):
                if size == 1:
                    grad = grad.sum(axis=i, keepdims=True)
                else:
                    break

            return grad

        depends_on.append(Dependency(b, grad_fn2))

    return Tensor(data, requires_grad, depends_on)


def tensor_mul(a: Tensor, b: Tensor) -> Tensor:
    """
    Return the multiplication of two Tensor. Broadcasting is allowed.
    :param a: left tensor
    :param b: right tensor
    :return: a tensor representing the sum of the two tensor
    """
    data = a.data * b.data
    requires_grad = a.requires_grad or b.requires_grad
    depends_on: List[Dependency] = []

    if a.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            grad = grad * b.data

            # added ndim explicitly
            ndims_added = grad.ndim - a.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            # no added ndim but still broadcasting through the dims which have size 1
            for i, size in enumerate(a.shape):
                if size == 1:
                    grad = grad.sum(axis=i, keepdims=True)
                else:
                    break

            return grad

        depends_on.append(Dependency(a, grad_fn1))

    if b.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            grad = grad * a.data

            # added ndim explicitly
            ndims_added = grad.ndim - b.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            # no added ndim but still broadcasting through the dims which have size 1
            for i, size in enumerate(b.shape):
                if size == 1:
                    grad = grad.sum(axis=i, keepdims=True)
                else:
                    break

            return grad

        depends_on.append(Dependency(b, grad_fn2))

    return Tensor(data, requires_grad, depends_on)