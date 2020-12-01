
#利用图卷积邻居聚合（包括自环：renorm=True，默认为True)进行图节点分类
# coding=utf-8
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tf_geometric as tfg
import tensorflow as tf
from tensorflow import keras
from tf_geometric.utils.graph_utils import add_self_loop_edge

graph, (train_index, valid_index, test_index) = tfg.datasets.CoraDataset().load_data()
print(graph.x.shape, graph.edge_index.shape)#(2708, 1433) (2, 10556),边是有向的，是把原数据中的每条无向边首尾复制一次得到的
#例如连接节点5和节点8的边，在edge_index中就变成了[[5,8],[8,5]]
#若原数据中的边是有向的，那么在后续邻居聚合时只聚合被中心节点发出的边指向的邻居

num_classes = graph.y.max() + 1#7
drop_rate = 0.2

class GCN(tf.keras.Model):

    def __init__(self, rate, num_classes):
        super(GCN, self).__init__()
        self.rate = rate
        self.num_classes = num_classes
        self.gcn0 = tfg.layers.GCN(64, activation=tf.nn.relu)#attention_units * num_heads = 维度（64）
        self.gcn1 = tfg.layers.GCN(self.num_classes)#由于后续在训练时使用softmax_cross_entropy训练，此处不需要使用softmax激活
        self.dropout = keras.layers.Dropout(self.rate)

    def call(self, graph):
        h = graph.x
        h = self.dropout(h)
        h = self.gcn0([h, graph.edge_index, graph.edge_weight], cache=graph.cache)#每张图的cache是独有的
        h = self.dropout(h)
        h = self.gcn1([h, graph.edge_index, graph.edge_weight], cache=graph.cache)
        return h




def compute_loss(logits, mask_index, vars):
    masked_logits = tf.gather(logits, mask_index)#获取训练集元素在集中的位置
    masked_labels = tf.gather(graph.y, mask_index)
    losses = tf.nn.softmax_cross_entropy_with_logits(
        logits=masked_logits,
        labels=tf.one_hot(masked_labels, depth=num_classes)
    )

    kernel_vals = [var for var in vars if "kernel" in var.name]
    l2_losses = [tf.nn.l2_loss(kernel_var) for kernel_var in kernel_vals]
    #添加l2损失，这使得模型的权重参数趋近于0，避免过拟合（若l1损失则使部分权重直接等于0）

    return tf.reduce_mean(losses) + tf.add_n(l2_losses) * 5e-4


def evaluate(logits):#评估
    
    masked_logits = tf.gather(logits, test_index)#获取测试集对应元素
    masked_labels = tf.gather(graph.y, test_index)

    y_pred = tf.argmax(masked_logits, axis=-1, output_type=tf.int32)

    accuracy_m = keras.metrics.Accuracy()
    accuracy_m.update_state(masked_labels, y_pred)

    return accuracy_m.result().numpy()


optimizer = tf.keras.optimizers.Adam(learning_rate=5e-3)
gcn = GCN(drop_rate, num_classes)

def train_step():
    for step in range(2000):
        with tf.GradientTape() as tape:
            logits = gcn.call(graph)
            loss = compute_loss(logits, train_index, tape.watched_variables())

        vars = gcn.trainable_variables
        grads = tape.gradient(loss, vars)
        optimizer.apply_gradients(zip(grads, vars))

        if step % 20 == 0:
            accuracy = evaluate(logits)
            print("step = {}\tloss = {}\taccuracy = {}".format(step, loss, accuracy))

            