import unittest
from autograd.Tensor import Tensor


class TestTensorAdd(unittest.TestCase):

    def test_add1(self):
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([3, 4, 5], requires_grad=True)

        s = a - b
        grad = Tensor([3, 4, 8])
        s.backward(grad)

        assert a.grad.data.tolist() == [3, 4, 8]
        assert b.grad.data.tolist() == [-3, -4, -8]

    def test_add2(self):
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([[3, 4, 5], [3, 4, 5], [3, 4, 5]], requires_grad=True)

        s = a - b
        s.backward()

        assert a.grad.data.tolist() == [3, 3, 3]
        assert b.grad.data.tolist() == [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]

    def test_add3(self):
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([[3, 4, 5], [3, 4, 5], [3, 4, 5]], requires_grad=True)

        s = a - b
        grad = Tensor([[3, 4, 8], [3, 4, 8], [3, 4, 8]])
        s.backward(grad)

        assert a.grad.data.tolist() == [9, 12, 24]
        assert b.grad.data.tolist() == [[-3, -4, -8], [-3, -4, -8], [-3, -4, -8]]

    def test_sub4(self):
        a = Tensor([1, 2, 3], requires_grad=True)
        a -= 1
        assert a.data.tolist() == [0, 1, 2]