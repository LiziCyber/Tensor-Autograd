import unittest
from autograd.Tensor import Tensor


class TestTensorSum(unittest.TestCase):
    def test_sum_without_grad(self, requires_grad=True):
        a = Tensor([1, 2, 3], requires_grad=True)
        s = a.sum()

        s.backward()

        assert a.grad.data.tolist() == [1, 1, 1]

    def test_sum_with_grad(self):
        a = Tensor([1, 2, 3], requires_grad=True)
        grad = Tensor([3, 4, 8])
        s = a.sum()

        s.backward(grad)

        assert a.grad.data.tolist() == [3, 4, 8]

