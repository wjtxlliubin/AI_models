from .Node import Node

class Linear(Node):
    def __init__(self, x: None, weigth: None, bias: None, name=None, is_trainable=False):
        Node.__init__(self, [x, weigth, bias], name=name, is_trainable=False)

    def forward(self):
        k, x, b = self.inputs[1], self.inputs[0], self.inputs[2]
        self.value = k.value * x.value + b.value

    def backward(self):
        k, x, b = self.inputs[1], self.inputs[0], self.inputs[2]

        for n in self.outputs:
            grad_cost = n.gradients[self]

            self.gradients[k] = grad_cost * x.value

            self.gradients[x] = grad_cost * k.value

            self.gradients[b] = grad_cost * 1