import numpy as np
from ..network.Node import Node

class Sigmoid(Node):
    def __init__(self, x, name=None, is_trainable=False):
        Node.__init__(self, [x], name=name, is_trainable=False)
        self.x = self.inputs[0]

    def _sigmoid(self, x):
        return 1. / (1 + np.exp(-1 * x))

    def forward(self):
        self.value = self._sigmoid(self.x.value)

    def partial(self):
        return self._sigmoid(self.x.value) * (1 - self._sigmoid(self.x.value))

    def backward(self):
        self.gradients[self.x] = 0

        for n in self.outputs:
            grad_cost = n.gradients[self]
            self.gradients[self.x] += grad_cost * self.partial()