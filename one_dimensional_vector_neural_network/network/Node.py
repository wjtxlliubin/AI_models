class Node:
    def __init__(self, inputs=[], name=None, is_trainable=False):
        self.inputs = inputs
        self.outputs = []
        self.name = name
        self.is_trainable = is_trainable

        for n in self.inputs:
            n.outputs.append(self)

        self.value = None

        self.gradients = {}

    def forward(self):
        pass

    def backward(self):
        pass

    def __repr__(self):
        return '{}'.format(self.name)