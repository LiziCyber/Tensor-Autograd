from autograd.Module import Module


class SGD:
    def __init__(self, lr=0.01) -> None:
        self.lr = lr

    def step(self, model: Module) -> None:
        for parameter in model.parameters():
            parameter -= parameter.grad * self.lr
