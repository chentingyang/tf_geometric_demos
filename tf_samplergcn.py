#利用图卷积进行图节点分类,但在采样邻居节点时只选取部分邻居
# coding=utf-8
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tf_geometric as tfg
import tensorflow as tf
from tensorflow import keras
from tf_geometric.utils.graph_utils import add_self_loop_edge
from tf_geometric.utils.graph_utils import RandomNeighborSampler#邻居随机采样

graph, (train_index, valid_index, test_index) = tfg.datasets.CoraDataset().load_data()
print(graph.x.shape, graph.edge_index.shape)#(2708, 1433) (2, 10556)
num_classes = graph.y.max() + 1#7
drop_rate = 0.2

neighbor_sampler = RandomNeighborSampler(graph.edge_index)
#邻居节点随机采样，它有两个参数，一个是k,规定采样邻居个数，若邻居个数不足k个的，则对邻居进行多次采样直至k个样本
#另一个参数是ratio,它控制采样邻居的比例，两个参数只需提供其中一个
#两层GCNloss为0.45，acc为0.81

class GCN(tf.keras.Model):#对照组，看一个4层GCN的训练效果, loss = 1.1754034757614136 accuracy = 0.503000020980835

    def __init__(self, rate, num_classes):
        super(GCN, self).__init__()
        self.rate = rate
        self.num_classes = num_classes
        self.gcn0 = tfg.layers.GCN(256, activation = tf.nn.relu)
        self.gcn1 = tfg.layers.GCN(128, activation=tf.nn.relu)#attention_units * num_heads = 维度（64）
        self.gcn2 = tfg.layers.GCN(64, activation=tf.nn.relu)
        self.gcn3 = tfg.layers.GCN(self.num_classes)
        self.dropout = keras.layers.Dropout(self.rate)

    def call(self, graph):
        h = graph.x
        h = self.dropout(h)
        h = self.gcn0([h, graph.edge_index, graph.edge_weight], cache=graph.cache)#GCN操作格式
        h = self.dropout(h)
        h = self.gcn1([h, graph.edge_index, graph.edge_weight], cache=graph.cache)
        h = self.dropout(h)
        h = self.gcn2([h, graph.edge_index, graph.edge_weight], cache=graph.cache)
        h = self.dropout(h)
        h = self.gcn3([h, graph.edge_index, graph.edge_weight], cache=graph.cache)
        return h

class KSamplerGCN(tf.keras.Model):#一个指定邻居个数的采样GCN(4层升采样),loss = 0.4289660155773163    accuracy = 0.6480000019073486

    def __init__(self, rate, num_classes):
        super(KSamplerGCN, self).__init__()
        self.rate = rate
        self.num_classes = num_classes
        self.gcn0 = tfg.layers.GCN(256, activation = tf.nn.relu)
        self.gcn1 = tfg.layers.GCN(128, activation=tf.nn.relu)
        self.gcn2 = tfg.layers.GCN(64, activation=tf.nn.relu)
        self.gcn3 = tfg.layers.GCN(self.num_classes)
        self.dropout = keras.layers.Dropout(self.rate)

    def call(self, graph):
        h = graph.x
        h = self.dropout(h)
        sampled_edge_index_0, sampled_edge_weight_0 = neighbor_sampler.sample(k=6)#第一层升采样
        h = self.gcn0([h, sampled_edge_index_0, sampled_edge_weight_0], cache=graph.cache)

        h = self.dropout(h)
        sampled_edge_index_1, sampled_edge_weight_1 = neighbor_sampler.sample(k=6)
        h = self.gcn1([h, sampled_edge_index_1, sampled_edge_weight_1], cache=graph.cache)

        h = self.dropout(h)
        sampled_edge_index_2, sampled_edge_weight_2 = neighbor_sampler.sample(k=5)
        h = self.gcn2([h, sampled_edge_index_2, sampled_edge_weight_2], cache=graph.cache)

        h = self.dropout(h)
        sampled_edge_index_3, sampled_edge_weight_3 = neighbor_sampler.sample(k=5)
        h = self.gcn3([h, sampled_edge_index_3, sampled_edge_weight_3], cache=graph.cache)

        return h

class KRandSamplerGCN(tf.keras.Model):#一个指定邻居个数的采样GCN.采样时采用随机采样技术，但是每个节点的采样数都为图中节点的平均邻居数
#这样对邻居多的节点限制聚合，对邻居少的节点突出邻居特征，达到loss=0.2, acc=0.75，效果最好


    def __init__(self, rate, num_classes):
        super(KRandSamplerGCN, self).__init__()
        self.rate = rate
        self.num_classes = num_classes
        self.gcn0 = tfg.layers.GCN(256, activation = tf.nn.relu)
        self.gcn1 = tfg.layers.GCN(128, activation=tf.nn.relu)
        self.gcn2 = tfg.layers.GCN(64, activation=tf.nn.relu)
        self.gcn3 = tfg.layers.GCN(self.num_classes)
        self.dropout = keras.layers.Dropout(self.rate)

    def call(self, graph):
        h = graph.x
        h = self.dropout(h)
        sampled_edge_index_0, sampled_edge_weight_0 = neighbor_sampler.sample(k=4)
        h = self.gcn0([h, sampled_edge_index_0, sampled_edge_weight_0], cache=graph.cache)

        h = self.dropout(h)
        sampled_edge_index_1, sampled_edge_weight_1 = neighbor_sampler.sample(k=4)
        h = self.gcn1([h, sampled_edge_index_1, sampled_edge_weight_1], cache=graph.cache)

        h = self.dropout(h)
        sampled_edge_index_2, sampled_edge_weight_2 = neighbor_sampler.sample(k=4)
        h = self.gcn2([h, sampled_edge_index_2, sampled_edge_weight_2], cache=graph.cache)

        h = self.dropout(h)
        sampled_edge_index_3, sampled_edge_weight_3 = neighbor_sampler.sample(k=4)
        h = self.gcn3([h, sampled_edge_index_3, sampled_edge_weight_3], cache=graph.cache)

        return h

class RatioSamplerGCN(tf.keras.Model):#一个指定采样比例的邻居降采样GCN,loss = 0.4289924204349518  accuracy = 0.6389999985694885

    def __init__(self, rate, num_classes):
        super(RatioSamplerGCN, self).__init__()
        self.rate = rate
        self.num_classes = num_classes
        self.gcn0 = tfg.layers.GCN(256, activation = tf.nn.relu)
        self.gcn1 = tfg.layers.GCN(128, activation=tf.nn.relu)）
        self.gcn2 = tfg.layers.GCN(64, activation=tf.nn.relu)
        self.gcn3 = tfg.layers.GCN(self.num_classes)
        self.dropout = keras.layers.Dropout(self.rate)

    def call(self, graph):
        h = graph.x
        h = self.dropout(h)
        sampled_edge_index_0, sampled_edge_weight_0 = neighbor_sampler.sample(ratio=1)
        h = self.gcn0([h, sampled_edge_index_0, sampled_edge_weight_0], cache=graph.cache)

        h = self.dropout(h)
        sampled_edge_index_1, sampled_edge_weight_1 = neighbor_sampler.sample(ratio=0.8)
        h = self.gcn1([h, sampled_edge_index_1, sampled_edge_weight_1], cache=graph.cache)

        h = self.dropout(h)
        sampled_edge_index_2, sampled_edge_weight_2 = neighbor_sampler.sample(ratio=0.6)
        h = self.gcn2([h, sampled_edge_index_2], cache=graph.cache)
        h = self.gcn2([h, sampled_edge_index_2], cache=graph.cache)

        h = self.dropout(h)
        sampled_edge_index_3, sampled_edge_weight_3 = neighbor_sampler.sample(ratio=0.5)
        h = self.gcn3([h, sampled_edge_index_3, sampled_edge_weight_3], cache=graph.cache)
        
        return h



def compute_loss(logits, mask_index, vars):
    masked_logits = tf.gather(logits, mask_index)#获取所求元素在集中的位置
    masked_labels = tf.gather(graph.y, mask_index)
    losses = tf.nn.softmax_cross_entropy_with_logits(
        logits=masked_logits,
        labels=tf.one_hot(masked_labels, depth=num_classes)
    )

    kernel_vals = [var for var in vars if "kernel" in var.name]
    l2_losses = [tf.nn.l2_loss(kernel_var) for kernel_var in kernel_vals]#添加l2损失

    return tf.reduce_mean(losses) + tf.add_n(l2_losses) * 5e-4


def evaluate(logits):#评估
    
    masked_logits = tf.gather(logits, test_index)#获取测试集对应元素
    masked_labels = tf.gather(graph.y, test_index)

    y_pred = tf.argmax(masked_logits, axis=-1, output_type=tf.int32)

    accuracy_m = keras.metrics.Accuracy()
    accuracy_m.update_state(masked_labels, y_pred)

    return accuracy_m.result().numpy()


optimizer = tf.keras.optimizers.Adam(learning_rate=5e-3)
kgcn = KSamplerGCN(drop_rate, num_classes)
rgcn = RatioSamplerGCN(drop_rate, num_classes)
krgcn = KRandSamplerGCN(drop_rate, num_classes)
gcn = GCN(drop_rate, num_classes)

def train_step_gcn():
    for step in range(800):
        with tf.GradientTape() as tape:
            logits = gcn.call(graph)
            loss = compute_loss(logits, train_index, tape.watched_variables())

        vars = gcn.trainable_variables
        grads = tape.gradient(loss, vars)
        optimizer.apply_gradients(zip(grads, vars))

        if step % 20 == 0:
            accuracy = evaluate(logits)
            print("step = {}\tloss = {}\taccuracy = {}".format(step, loss, accuracy))

def train_step_kgcn():
    for step in range(800):
        with tf.GradientTape() as tape:
            logits = kgcn.call(graph)
            loss = compute_loss(logits, train_index, tape.watched_variables())

        vars = kgcn.trainable_variables
        grads = tape.gradient(loss, vars)
        optimizer.apply_gradients(zip(grads, vars))

        if step % 20 == 0:
            accuracy = evaluate(logits)
            print("step = {}\tloss = {}\taccuracy = {}".format(step, loss, accuracy))

def train_step_krgcn():
    for step in range(800):
        with tf.GradientTape() as tape:
            logits = krgcn.call(graph)
            loss = compute_loss(logits, train_index, tape.watched_variables())

        vars = krgcn.trainable_variables
        grads = tape.gradient(loss, vars)
        optimizer.apply_gradients(zip(grads, vars))

        if step % 20 == 0:
            accuracy = evaluate(logits)
            print("step = {}\tloss = {}\taccuracy = {}".format(step, loss, accuracy))



def train_step_rgcn():
    for step in range(800):
        with tf.GradientTape() as tape:
            logits = rgcn.call(graph)
            loss = compute_loss(logits, train_index, tape.watched_variables())

        vars = rgcn.trainable_variables
        grads = tape.gradient(loss, vars)
        optimizer.apply_gradients(zip(grads, vars))

        if step % 20 == 0:
            accuracy = evaluate(logits)
            print("step = {}\tloss = {}\taccuracy = {}".format(step, loss, accuracy))


train_step_rgcn()
