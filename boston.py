from one_dimensional_vector_neural_network.network.Placeholder import Placeholder
from one_dimensional_vector_neural_network.network.Linear import Linear
from one_dimensional_vector_neural_network.loss.L2_LOSS import L2_LOSS
from one_dimensional_vector_neural_network.activationFunction.Relu import Relu
from one_dimensional_vector_neural_network.graph.Graph import graph
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm


def forward_and_backward(graph_order, monitor=False):
    # 整体的参数就更新了一次
    for node in graph_order:
        if monitor:
            print('forward computing node: {}'.format(node))
        node.forward()

    for node in graph_order[::-1]:
        if monitor:
            print('backward computing node: {}'.format(node))
        node.backward()

def optimizer(graph, learning_rate=1e-2):
    for t in graph:
        if t.is_trainable:
            t.value += -1 * t.gradients[t] * learning_rate

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
X_ = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
y_ = raw_df.values[1::2, 2]

X_rm = X_[:, 5]

w1_, b1_ = np.random.normal(), np.random.normal()
w2_, b2_ = np.random.normal(), np.random.normal()
w3_, b3_ = np.random.normal(), np.random.normal()

X, y = Placeholder(name='X'), Placeholder(name='y')
w1, b1 = Placeholder(name='w1', is_trainable=True), Placeholder(name='b1', is_trainable=True)
w2, b2 = Placeholder(name='w2', is_trainable=True), Placeholder(name='b2', is_trainable=True)
w3, b3 = Placeholder(name='w3', is_trainable=True), Placeholder(name='b3', is_trainable=True)
# build model

output1 = Linear(X, w1, b1, name='linear_01')
# output2 = Sigmoid(output1, name='sigmoid')
output2 = Relu(output1, name='relu')
y_hat = Linear(output2, w2, b2, name='linear_02')
cost = L2_LOSS(y, y_hat, name='loss')

feed_dict = {
    X: X_rm,
    y: y_,
    w1: w1_,
    w2: w2_,
    b1: b1_,
    b2: b2_,
}

graph_sort = graph.node_compting_sort(feed_dict)

epoch = 100

learning_rate = 1e-2
batch_num = 100

losses = []

for e in tqdm(range(epoch)):

    batch_loss = 0

    for b in range(batch_num):
        index = np.random.choice(range(len(X_rm)))
        X.value = X_rm[index]
        y.value = y_[index]

        forward_and_backward(graph_sort, monitor=False)

        optimizer(graph_sort, learning_rate=learning_rate)
        # sgd stocastic gradient descent

        batch_loss += cost.value

    losses.append(batch_loss / batch_num)

print(losses)