import unittest
from autograd.Tensor import Tensor


class TestTensorSlice(unittest.TestCase):

    def test1(self):
        a = Tensor([1, 2, 3], requires_grad=True)
        assert a[0].data.tolist() == 1