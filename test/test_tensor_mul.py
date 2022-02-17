import unittest
from autograd.Tensor import Tensor


class TestTensorAdd(unittest.TestCase):
    def test_mul1(self, requires_grad=True):
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([3, 4, 5])

        s = a * b
        s.backward()

        assert a.grad.data.tolist() == [3, 4, 5]

    def test_mul2(self):
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([3, 4, 5], requires_grad=True)

        s = a * b
        grad = Tensor([3, 4, 8])
        s.backward(grad)

        assert a.grad.data.tolist() == [9, 16, 40]
        assert b.grad.data.tolist() == [3, 8, 24]

    def test_mul3(self):
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([[3, 4, 5], [3, 4, 5], [3, 4, 5]], requires_grad=True)

        s = a * b
        s.backward()

        assert a.grad.data.tolist() == [9, 12, 15]
        assert b.grad.data.tolist() == [[1, 2, 3], [1, 2, 3], [1, 2, 3]]

    def test_mul4(self):
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([[3, 4, 5], [3, 4, 5], [3, 4, 5]], requires_grad=True)

        s = a * b
        grad = Tensor([[2, 2, 2], [2, 2, 2], [2, 2, 2]])
        s.backward(grad)

        assert a.grad.data.tolist() == [18, 24, 30]
        assert b.grad.data.tolist() == [[2, 4, 6], [2, 4, 6], [2, 4, 6]]
