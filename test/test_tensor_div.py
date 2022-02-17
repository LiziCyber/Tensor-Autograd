import unittest
from autograd.Tensor import Tensor
import numpy as np


class TestTensorDiv(unittest.TestCase):
    def test_div1(self, requires_grad=True):
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([3, 4, 5], requires_grad=True)

        s = a / b
        s.backward()

        np.testing.assert_almost_equal(a.grad.data, 1/b.data)
        np.testing.assert_almost_equal(b.grad.data, -a.data/(b.data * b.data))

    def test_div2(self):
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor(10)

        s = a / b
        s.backward()

        np.testing.assert_almost_equal(a.grad.data, 1 / b.data)

    def test_div3(self):
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor(10)

        s = b / a
        s.backward()

        np.testing.assert_almost_equal(a.grad.data, -b.data / (a.data * a.data))

    def test_add4(self):
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([[3, 4, 5], [3, 4, 5], [3, 4, 5]], requires_grad=True)

        s = a + b
        grad = Tensor([[3, 4, 8], [3, 4, 8], [3, 4, 8]])
        s.backward(grad)

        assert a.grad.data.tolist() == [9, 12, 24]
        assert b.grad.data.tolist() == [[3, 4, 8], [3, 4, 8], [3, 4, 8]]
