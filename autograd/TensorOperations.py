from autograd.Tensor import Tensor, Dependency
from typing import List
import numpy as np


def _tensor_sum(t: Tensor) -> Tensor:
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


def _tensor_add(a: Tensor, b: Tensor) -> Tensor:
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


def _tensor_mul(a: Tensor, b: Tensor) -> Tensor:
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


def _tensor_neg(a: Tensor) -> Tensor:
    """
    return -a as a tensor
    """
    data = -a.data
    requires_grad = a.requires_grad
    if requires_grad:
        depends_on = [Dependency(a, lambda x: -x)]
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)


def _tensor_sub(a: Tensor, b: Tensor) -> Tensor:
    """
    Return the result a-b as a tensor
    """
    return _tensor_add(a, _tensor_neg(b))


def _tensor_pow(a: Tensor, b: Tensor) -> Tensor:
    """
    Not finished yet
    """
    data = a.data ** b.data
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
