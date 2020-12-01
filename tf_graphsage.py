# coding=utf-8
import os
#利用采样图卷积graphsage实现图节点分类，其核心是随机采样邻居节点RandomNeighborSampler+邻居节点（不包自环）特征聚合，与中心节点特征并行拼接
#这样缩小了卷积感受野，增大了中心节点的特征权重，避免节点经多次卷积后收到全图节点的影响
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tf_geometric as tfg
import tensorflow as tf
from tensorflow import keras
from tf_geometric.utils.graph_utils import RandomNeighborSampler#邻居随机采样


graph, (train_index, valid_index, test_index) = tfg.datasets.CoraDataset().load_data()

num_classes = graph.y.max() + 1
drop_rate = 0.2

neighbor_sampler = RandomNeighborSampler(graph.edge_index)#邻居节点随机采样



class MeanGraphSage(tf.keras.Model):#邻居平均的采样图卷积，先对邻居进行随机采样，再对邻居特征向量取均值加和，在此基础上，进行维度变换（全连层）
#同时，中心节点也进行维度变换，二者维度变换到MeanPoolGraphSage的维度参数上，再将二者进行拼接，所以每层的输出维度是输入参数的2倍
    def __init__(self, rate, num_classes):#(两个graphsage层+一个图卷积层)
        super(MeanGraphSage, self).__init__()
        self.rate = rate
        self.num_classes = num_classes
        self.gsg0 = tfg.layers.MeanGraphSage(64, activation=tf.nn.relu)
        self.gsg1 = tfg.layers.MeanGraphSage(32, activation=tf.nn.relu)
        self.gcn = tfg.layers.GCN(self.num_classes)#由于num_classes为奇数，所以后面需要添加图卷积层
        self.dropout = keras.layers.Dropout(self.rate)

    def call(self, graph):
        h = graph.x

        h = self.dropout(h)
        sampled_edge_index_0, sampled_edge_weight_0 = neighbor_sampler.sample(k=3)
        h = self.gsg0([h, sampled_edge_index_0, sampled_edge_weight_0], cache=graph.cache)#(2708. 64*2)

        h = self.dropout(h)
        sampled_edge_index_1, sampled_edge_weight_1 = neighbor_sampler.sample(ratio=0.5)
        h = self.gsg1([h, sampled_edge_index_1, sampled_edge_weight_1], cache=graph.cache)#(2708, 32*2)

        h = self.dropout(h)
        h = self.gcn([h, graph.edge_index, graph.edge_weight], cache=graph.cache)


        return h

class MaxPoolGraphSage(tf.keras.Model):#最大池化的采样图卷积，先对邻居进行随机采样，再对经过全连层的邻居特征向量作最大池化，在此基础上，进行维度变换
#同时，中心节点也进行维度变换，二者维度变换到MaxPoolGraphSage的维度参数上，再将二者进行拼接，所以每层的输出维度是输入参数的2倍
    def __init__(self, rate, num_classes):#(两个graphsage层+一个图卷积层)
        super(MaxPoolGraphSage, self).__init__()
        self.rate = rate
        self.num_classes = num_classes
        self.gsg0 = tfg.layers.MaxPoolGraphSage(64, activation=tf.nn.relu)
        self.gsg1 = tfg.layers.MaxPoolGraphSage(32, activation=tf.nn.relu)
        self.gcn = tfg.layers.GCN(self.num_classes)#由于num_classes为奇数，所以后面需要添加图卷积层
        self.dropout = keras.layers.Dropout(self.rate)

    def call(self, graph):
        h = graph.x

        h = self.dropout(h)
        sampled_edge_index_0, sampled_edge_weight_0 = neighbor_sampler.sample(k=3)
        h = self.gsg0([h, sampled_edge_index_0, sampled_edge_weight_0], cache=graph.cache)#(2708. 64*2)

        h = self.dropout(h)
        sampled_edge_index_1, sampled_edge_weight_1 = neighbor_sampler.sample(ratio=0.5)
        h = self.gsg1([h, sampled_edge_index_1, sampled_edge_weight_1], cache=graph.cache)#(2708, 32*2)

        h = self.dropout(h)
        h = self.gcn([h, graph.edge_index, graph.edge_weight], cache=graph.cache)


        return h

class LSTMGraphSage(tf.keras.Model):#lstm的采样图卷积，先对邻居进行随机采样，再将邻居节点打乱后当做序列输入到lstm中进行维度变换
#同时，中心节点也进行维度变换，二者维度变换到MaxPoolGraphSage的维度参数上，再将二者进行拼接，所以每层的输出维度是输入参数的2倍
    def __init__(self, rate, num_classes):#(两个graphsage层+一个图卷积层)
        super(LSTMGraphSage, self).__init__()
        self.rate = rate
        self.num_classes = num_classes
        self.gsg0 = tfg.layers.LSTMGraphSage(64, activation=tf.nn.relu)
        self.gsg1 = tfg.layers.LSTMGraphSage(32, activation=tf.nn.relu)
        self.gcn = tfg.layers.GCN(self.num_classes)#由于num_classes为奇数，所以后面需要添加图卷积层
        self.dropout = keras.layers.Dropout(self.rate)

    def call(self, graph):
        h = graph.x

        h = self.dropout(h)
        sampled_edge_index_0, sampled_edge_weight_0 = neighbor_sampler.sample(k=3)
        h = self.gsg0([h, sampled_edge_index_0, sampled_edge_weight_0], cache=graph.cache)#(2708. 64*2)

        h = self.dropout(h)
        sampled_edge_index_1, sampled_edge_weight_1 = neighbor_sampler.sample(ratio=0.5)
        h = self.gsg1([h, sampled_edge_index_1, sampled_edge_weight_1], cache=graph.cache)#(2708, 32*2)

        h = self.dropout(h)
        h = self.gcn([h, graph.edge_index, graph.edge_weight], cache=graph.cache)


        return h

class GraphSage(tf.keras.Model):#对照，只进行节点聚合而不随机采样的graphsage
    def __init__(self, rate, num_classes):#(两个graphsage层+一个图卷积层)
        super(GraphSage, self).__init__()
        self.rate = rate
        self.num_classes = num_classes
        self.gsg0 = tfg.layers.MeanGraphSage(64, activation=tf.nn.relu)
        self.gsg1 = tfg.layers.MeanGraphSage(32, activation=tf.nn.relu)
        self.gcn = tfg.layers.GCN(self.num_classes)#由于num_classes为奇数，所以后面需要添加图卷积层
        self.dropout = keras.layers.Dropout(self.rate)

    def call(self, graph):
        h = graph.x

        h = self.dropout(h)
        h = self.gsg0([h, graph.edge_index, tf.ones([graph.edge_index.shape[1]])], cache=graph.cache)#(2708. 64*2)， 节点权重均设为1

        h = self.dropout(h)
        sampled_edge_index_1, sampled_edge_weight_1 = neighbor_sampler.sample(ratio=0.5)
        h = self.gsg1([h, graph.edge_index, tf.ones([graph.edge_index.shape[1]])], cache=graph.cache)#(2708, 32*2)

        h = self.dropout(h)
        h = self.gcn([h, graph.edge_index, graph.edge_weight], cache=graph.cache)


        return h




def compute_loss(logits, mask_index, vars):
    masked_logits = tf.gather(logits, mask_index)
    masked_labels = tf.gather(graph.y, mask_index)
    losses = tf.nn.softmax_cross_entropy_with_logits(
        logits=masked_logits,
        labels=tf.one_hot(masked_labels, depth=num_classes)
    )

    kernel_vals = [var for var in vars if "kernel" in var.name]#筛选出网络参数层
    l2_losses = [tf.nn.l2_loss(kernel_var) for kernel_var in kernel_vals]#添加参数权重l2loss,这样在优化过程中部分参数会被置零，增强泛化能力

    return tf.reduce_mean(losses) + tf.add_n(l2_losses) * 5e-4



def evaluate(logits):#评估
    
    masked_logits = tf.gather(logits, test_index)#获取测试集对应元素
    masked_labels = tf.gather(graph.y, test_index)

    y_pred = tf.argmax(masked_logits, axis=-1, output_type=tf.int32)

    accuracy_m = keras.metrics.Accuracy()
    accuracy_m.update_state(masked_labels, y_pred)

    return accuracy_m.result().numpy()


optimizer = tf.keras.optimizers.Adam(learning_rate=5e-3)
gsg = MeanGraphSage(drop_rate, num_classes)
mgsg = MaxPoolGraphSage(drop_rate, num_classes)
lgsg = LSTMGraphSage(drop_rate, num_classes) 
sg = GraphSage(drop_rate, num_classes)

def train_step():
    for step in range(800):
        with tf.GradientTape() as tape:
            logits = gsg.call(graph)
            loss = compute_loss(logits, train_index, tape.watched_variables())

        vars = gsg.trainable_variables
        grads = tape.gradient(loss, vars)
        optimizer.apply_gradients(zip(grads, vars))

        if step % 20 == 0:
            accuracy = evaluate(logits)
            print("step = {}\tloss = {}\taccuracy = {}".format(step, loss, accuracy))

def mp_train_step():
    for step in range(800):
        with tf.GradientTape() as tape:
            logits = mgsg.call(graph)
            loss = compute_loss(logits, train_index, tape.watched_variables())

        vars = mgsg.trainable_variables
        grads = tape.gradient(loss, vars)
        optimizer.apply_gradients(zip(grads, vars))

        if step % 20 == 0:
            accuracy = evaluate(logits)
            print("step = {}\tloss = {}\taccuracy = {}".format(step, loss, accuracy))

def lstm_train_step():
    for step in range(800):
        with tf.GradientTape() as tape:
            logits = lgsg.call(graph)
            loss = compute_loss(logits, train_index, tape.watched_variables())

        vars = lgsg.trainable_variables
        grads = tape.gradient(loss, vars)
        optimizer.apply_gradients(zip(grads, vars))

        if step % 20 == 0:
            accuracy = evaluate(logits)
            print("step = {}\tloss = {}\taccuracy = {}".format(step, loss, accuracy))

def sg_train_step():
    for step in range(800):
        with tf.GradientTape() as tape:
            logits = sg.call(graph)
            loss = compute_loss(logits, train_index, tape.watched_variables())

        vars = sg.trainable_variables
        grads = tape.gradient(loss, vars)
        optimizer.apply_gradients(zip(grads, vars))

        if step % 20 == 0:
            accuracy = evaluate(logits)
            print("step = {}\tloss = {}\taccuracy = {}".format(step, loss, accuracy))

train_step()
mp_train_step()
lstm_train_step()
sg_train_step()