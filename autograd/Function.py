import numpy as np
from autograd.Tensor import Tensor, Dependency
from typing import Optional


def exp(x: Tensor) -> Tensor:
    data = np.exp(x.data)
    requires_grad = x.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * np.exp(x.data)

        depends_on = [Dependency(x, grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)


def log(x: Tensor) -> Tensor:
    data = np.log(x.data)
    requires_grad = x.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad / x.data

        depends_on = [Dependency(x, grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)


def logistic(x: Tensor) -> Tensor:
    temp = np.exp(x.data)
    data = 1 / (1 + np.exp(-temp))
    requires_grad = x.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * data * (1 - data)

        depends_on = [Dependency(x, grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)


def tanh(x: Tensor) -> Tensor:
    data = np.tanh(x.data)
    requires_grad = x.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * (1 - data * data)

        depends_on = [Dependency(x, grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)


def arctan(x: Tensor) -> Tensor:
    data = np.arctan(x.data)
    requires_grad = x.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad / (1 + x.data) ** 2

        depends_on = [Dependency(x, grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)


def relu(x: Tensor) -> Tensor:
    data = np.maximum(x.data, 0)
    requires_grad = x.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * (x.data >= 0)

        depends_on = [Dependency(x, grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)


def threshold(x: Tensor, thr: float, val: float) -> Tensor:
    data = x.data
    data[data >= thr] = val
    requires_grad = x.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * (x.data >= thr)

        depends_on = [Dependency(x, grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)


def softmax(x: Tensor) -> Tensor:
    temp = np.exp(x.data)
    data = temp / temp.sum()
    requires_grad = x.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return temp * (1 - temp.sum())

        depends_on = [Dependency(x, grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)


def linear(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    return (x @ weight) + (bias or 0)


def dropout(x: Tensor, p: float, training: bool = True) -> Tensor:
    assert 0 <= p <= 1, "dropout probability has to be between 0 and 1, " "but got {}".format(p)

    if training:
        pos = np.random.binomial(np.ones_like(x.data), p)
        data = x.data
        data[pos] = 0
    else:
        data = x.data

    requires_grad = x.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            if training:
                grad[pos] = 0
            return grad

        depends_on = [Dependency(x, grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)


def l1_loss(x: Tensor, y: Tensor) -> Tensor:
    return abs(x - y).mean()


def mse_loss(x: Tensor, y: Tensor) -> Tensor:
    return ((x - y) * (x - y)).mean()


def binary_cross_entropy(x: Tensor, y: Tensor) -> Tensor:
    return -(y * log(x) + (1 - y) * log(1 - x)).mean()


def cross_entropy(x: Tensor, y: Tensor) -> Tensor:
    return -(y*log(x)).mean()
