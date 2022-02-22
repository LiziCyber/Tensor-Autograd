from autograd.Module import Module
import numpy as np


class SGD:
    def __init__(self, model,
                 lr=0.01,
                 momentum=0,
                 dampening=0,
                 weight_decay=0,
                 nesterov=False) -> None:
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.b = []
        self.g = []

    def step(self) -> None:
        for t, parameter in enumerate(self.model.parameters()):
            grad = parameter.grad

            if self.weight_decay != 0:
                grad += parameter.data * self.weight_decay

            if self.momentum != 0:
                if len(self.b) > t:
                    self.b[t] = self.momentum * self.b[t-1] + (1 - self.dampening) * grad
                else:
                    self.b.append(grad)

                if self.nesterov:
                    if len(self.g) > t:
                        grad = self.g[t - 1] + self.momentum * self.b
                        self.g[t] = grad
                    else:
                        self.g.append(grad)
                else:
                    grad = self.b[t]

            parameter.data -= grad * self.lr


class Adam:
    def __init__(self, model,
                 lr=0.001,
                 betas=(0.9, 0.999),
                 eps=1e-08,
                 weight_decay=0,
                 amsgrad=False) -> None:
        self.model = model
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad

    def step(self) -> None:
        for parameter in self.model.parameters():
            parameter -= parameter.grad * self.lr


class RMSprop:
    def __init__(self, model,
                 lr=0.01,
                 alpha=0.99,
                 eps=1e-08,
                 weight_decay=0,
                 momentum=0,
                 centered=False) -> None:
        self.model = model
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.centered = centered
        self.v = []
        self.b = []
        self.gave = []

    def step(self) -> None:
        for t, parameter in enumerate(self.model.parameters()):
            grad = parameter.grad

            if self.weight_decay != 0:
                grad += parameter.data * self.weight_decay

            if len(self.v) > t:
                self.v[t] = self.alpha * self.v[t] + (1-self.alpha) * grad * grad
            else:
                self.v.append((1-self.alpha) * grad * grad)
            v_tide = self.v[t]

            if self.centered:
                if len(self.gave) > t:
                    self.gave[t] = self.alpha * self.gave[t] + (1 - self.alpha) * grad
                else:
                    self.gave.append((1 - self.alpha) * grad * grad)
                v_tide -= self.gave[t] * self.gave[t]

            if self.momentum > 0:
                if len(self.b) > t:
                    self.b[t] = self.momentum * self.b + grad / (np.sqrt(v_tide) + self.eps)
                else:
                    self.b.append(grad / (np.sqrt(v_tide) + self.eps))
                parameter.data -= self.lr * self.b[t]
            else:
                parameter.data -= self.lr * grad / (np.sqrt(v_tide) + self.eps)
