#利用GCN的雏形chebynet进行节点聚合分类
#每层chebynet对图节点进行k阶邻居的带权聚合，第k阶邻居的聚合系数由图的拉普拉斯矩阵的最大特征值，k阶拉普拉斯多项式共同决定
# coding=utf-8
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tf_geometric as tfg
import tensorflow as tf
from tensorflow import keras
from tf_geometric.utils.graph_utils import add_self_loop_edge
from tf_geometric.utils.graph_utils import LaplacianMaxEigenvalue

graph, (train_index, valid_index, test_index) = tfg.datasets.CoraDataset().load_data()
print(graph.x.shape, graph.edge_index.shape)#(2708, 1433) (2, 10556),边是有向的，是把原数据中的每条无向边首尾复制一次得到的
#例如连接节点5和节点8的边，在edge_index中就变成了[[5,8],[8,5]]

num_classes = graph.y.max() + 1#7
drop_rate = 0.2

graph_lambda_max = LaplacianMaxEigenvalue(graph.x, graph.edge_index, graph.edge_weight)#图的拉普拉斯矩阵的最大特征值

class ChebyNet(tf.keras.Model):

    def __init__(self, rate, num_classes):
        super(ChebyNet, self).__init__()
        self.rate = rate
        self.num_classes = num_classes
        self.cheby = tfg.layers.ChebyNet(64, K=3, lambda_max=graph_lambda_max(normalization_type='rw'))#聚合三阶邻居
        self.dropout = keras.layers.Dropout(self.rate)
        self.dense = keras.layers.Dense(self.num_classes)

    def call(self, graph):
        h = graph.x
        h = self.dropout(h)
        h = self.cheby([h, graph.edge_index, graph.edge_weight], cache=graph.cache)#每张图的cache是独有的
        h = self.dropout(h)
        h = self.dense(h)
        return h

def compute_loss(logits, mask_index, vars):
    masked_logits = tf.gather(logits, mask_index)
    masked_labels = tf.gather(graph.y, mask_index)

    losses = tf.nn.softmax_cross_entropy_with_logits(
        logits=masked_logits,
        labels=tf.one_hot(masked_labels, depth=num_classes)
    )

    kernel_vals = [var for var in vars if "kernel" in var.name]
    l2_losses = [tf.nn.l2_loss(kernel_var) for kernel_var in kernel_vals]#添加l2loss

    return tf.reduce_mean(losses) + tf.add_n(l2_losses) * 5e-5


def evaluate(mask):
    logits = cheby(graph)
    logits = tf.nn.log_softmax(logits, axis=1)
    masked_logits = tf.gather(logits, mask)
    masked_labels = tf.gather(graph.y, mask)

    y_pred = tf.argmax(masked_logits, axis=-1, output_type=tf.int32)
    accuracy_m = keras.metrics.Accuracy()
    accuracy_m.update_state(masked_labels, y_pred)
    return accuracy_m.result().numpy()


cheby = ChebyNet(drop_rate, num_classes)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

for step in range(1,101):#step = 100    loss = 0.018003862351179123 valid_acc = 0.7879999876022339  test_acc = 0.7919999957084656
    with tf.GradientTape() as tape:
        logits = cheby(graph)
        logits = tf.nn.log_softmax(logits,axis=1)
        loss = compute_loss(logits, train_index, tape.watched_variables())

    vars = tape.watched_variables()
    grads = tape.gradient(loss, vars)
    optimizer.apply_gradients(zip(grads, vars))

    valid_acc = evaluate(valid_index)
    test_acc = evaluate(test_index)

    print("step = {}\tloss = {}\tvalid_acc = {}\ttest_acc = {}".format(step, loss, valid_acc, test_acc))