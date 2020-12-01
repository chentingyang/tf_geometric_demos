#利用gcn作为编码器，在编码过程中（或之后）利用sag_pool进行节点筛选，再利用pool图的池化读出，最终实现图分类
# coding=utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tf_geometric as tfg
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# TU Datasets: https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
graph_dicts = tfg.datasets.TUDataset("NCI1").load_data()
print(len(graph_dicts))#4110张图


# Since a TU dataset may contain node_labels, node_attributes etc., each of which can be used as node features
# We process each graph as a dict and return a list of dict for graphs
# You can easily construct you Graph object with the data dict

num_node_labels = np.max([np.max(graph_dict["node_labels"]) for graph_dict in graph_dicts]) + 1#节点类别数
print(num_node_labels)#37


def convert_node_labels_to_one_hot(node_labels):#将节点类别转为one_hot编码
    num_nodes = len(node_labels)
    x = np.zeros([num_nodes, num_node_labels], dtype=np.float32)
    x[list(range(num_nodes)), node_labels] = 1.0
    return x


def construct_graph(graph_dict):#构建图，将节点类别的one_hot作为节点特征(num_nodes, 37)，将图的类别标签作为图的ground_truth(1,)
    return tfg.Graph(
        x=convert_node_labels_to_one_hot(graph_dict["node_labels"]),
        edge_index=graph_dict["edge_index"],
        y=graph_dict["graph_label"]  # graph_dict["graph_label"] is a list with one int element
    )


graphs = [construct_graph(graph_dict) for graph_dict in graph_dicts]

num_classes = np.max([graph.y[0] for graph in graphs]) + 1#图的类别, 2


train_graphs, test_graphs = train_test_split(graphs, test_size=0.1)


def create_graph_generator(graphs, batch_size, infinite=False, shuffle=False):#图的迭代器，每次输出银行一个batch_size的图
    while True:
        dataset = tf.data.Dataset.range(len(graphs))
        if shuffle:
            dataset = dataset.shuffle(2000)
        dataset = dataset.batch(batch_size)#每次输出一个batch的图索引

        for batch_graph_index in dataset:#batch_size
            batch_graph_list = [graphs[i] for i in batch_graph_index]

            batch_graph = tfg.BatchGraph.from_graphs(batch_graph_list)#输出一个batch的图
            yield batch_graph

        if not infinite:
            break
#输出的batch中有节点及特征，邻接矩阵，边的权重以及标识节点属于哪张图的索引

batch_size = 512


class SAGPool(keras.Model):#节点筛选sag_pool结构，不改变节点的维度，它是在子图级别上进行操作的，而不是在batch级别

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.score_gcn = tfg.layers.GCN(1)#打分层

    def call(self, inputs, training=None, mask=None):
        x, edge_index, edge_weight, node_graph_index = inputs#x为图的节点数据, node_graph_index表示节点属于哪张图
        node_score = self.score_gcn([x, edge_index, edge_weight])#获得节点重要性打分

        topk_node_index = tfg.nn.topk_pool(node_graph_index, node_score, ratio=0.5)#取每张图打分前50%的节点予以保留

        sampled_batch_graph = tfg.BatchGraph(
            x=x * tf.nn.tanh(node_score),
            edge_index=edge_index,
            node_graph_index=node_graph_index,
            edge_graph_index=None,
            edge_weight=edge_weight
        ).sample_new_graph_by_node_index(topk_node_index)#获得节点筛选后的新图(batch_size)

        return sampled_batch_graph.x, \
               sampled_batch_graph.edge_index, \
               sampled_batch_graph.edge_weight, \
               sampled_batch_graph.node_graph_index

#sag_pool_h的参数层
num_gcns = 3
gcns = [tfg.layers.GCN(128, activation=tf.nn.relu) for _ in range(num_gcns)]
sag_pools = [SAGPool() for _ in range(num_gcns)]#3层图卷积-sag_pool聚合层

mlp = keras.Sequential([
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(num_classes)
])


#sag_pool_g的参数层
gcnes = [tfg.layers.GCN(128, activation=tf.nn.relu) for _ in range(num_gcns)]
sag_pool = SAGPool()
mlpg = keras.Sequential([
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(num_classes)
])

# SAGPool_h
def forward(batch_graph, training=False):
    
    h = batch_graph.x#(batch中节点的总个数，37)
    edge_index = batch_graph.edge_index#(2, batch中边的总个数)
    edge_weight = batch_graph.edge_weight
    node_graph_index = batch_graph.node_graph_index#(节点总个数)，标识节点属于哪张图的索引，利用它将batch输入拆分成子图，
    #使得在sag_pool的过程中batch_graph.x变为每张子图的x,batch_graph.edge_index变为每张子图的edge_index

    outputs = []
    for i in range(num_gcns):#每一层的节点数约是上一层的一半
        h = gcns[i]([h, edge_index, edge_weight])#图卷积编码器
        h, edge_index, edge_weight, node_graph_index = sag_pools[i]([h, edge_index, edge_weight, node_graph_index])#sag_pool
        #h.shape(当前保留下来的节点数*0.5，128)
        
        output = tf.concat([
            tfg.nn.mean_pool(h, node_graph_index),
            tfg.nn.max_pool(h, node_graph_index)
        ], axis=-1)#每一层设一个读出层，采用mean_pool和max_pool两种方式将各子图的节点归一化再进行拼接（归一化依据就是node_graph_index）
        #在读出层中，每张图只保留一个由各节点的特征池化（mean_pool和max_pool)拼接聚合成的超级节点，最终输出(batch_size(num_graph), 128*2)
        outputs.append(output)

    h = tf.reduce_sum(tf.stack(outputs, axis=1), axis=1)
    #3个读出层输出加和(batch_size, 128*2)

    # Predict Graph Labels，图分类预测层
    h = mlp(h, training=training)
    return h

# SAGPool_g
def gforward(batch_graph, training=False):
    
    h = batch_graph.x#(batch中节点的总个数，37)
    edge_index = batch_graph.edge_index#(2, batch中边的总个数)
    edge_weight = batch_graph.edge_weight
    node_graph_index = batch_graph.node_graph_index#(节点总个数)，利用它将batch输入拆分成子图，
    #使得在sag_pool的过程中batch_graph.x变为每张子图的x,batch_graph.edge_index变为每张子图的edge_index
    
    outputs = []

    for i in range(num_gcns):#三层图卷积编码器
        h = gcnes[i]([h, edge_index, edge_weight])#图卷积，(节点总个数, 128)，图分类不需要指定图cache
        outputs.append(h)#获得三次图卷积分别的输出

    h = tf.concat(outputs, axis=-1)#(节点总个数, 128*3),三次图卷积输出的拼接
    
    h, edge_index, edge_weight, node_graph_index = sag_pool([h, edge_index, edge_weight, node_graph_index])#sag_pool,(节点总个数*0.5, 128*3)

    output = tf.concat([
            tfg.nn.mean_pool(h, node_graph_index),
            tfg.nn.max_pool(h, node_graph_index)                            
        ], axis=-1)#读出层, (batch_size, 128*6)

    # Predict Graph Labels，图分类预测层
    output = mlpg(output, training=training)
    return output

def evaluate():
    accuracy_m = keras.metrics.Accuracy()

    for test_batch_graph in create_graph_generator(test_graphs, batch_size, shuffle=False, infinite=False):
        logits = forward(test_batch_graph)
        preds = tf.argmax(logits, axis=-1)
        accuracy_m.update_state(test_batch_graph.y, preds)

    return accuracy_m.result().numpy()

def gevaluate():
    accuracy_m = keras.metrics.Accuracy()

    for test_batch_graph in create_graph_generator(test_graphs, batch_size, shuffle=False, infinite=False):
        logits = gforward(test_batch_graph)
        preds = tf.argmax(logits, axis=-1)
        accuracy_m.update_state(test_batch_graph.y, preds)

    return accuracy_m.result().numpy()


optimizer = tf.keras.optimizers.Adam(1e-3)

train_batch_generator = create_graph_generator(train_graphs, batch_size, shuffle=True, infinite=True)

#sag_pool_g_train
for step in range(20000):#step = 16740  loss = 0.30821800231933594  accuracy = 0.7712895274162292
    train_batch_graph = next(train_batch_generator)
    
    with tf.GradientTape() as tape:
        logits = gforward(train_batch_graph, training=True)
        losses = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits,
            labels=tf.one_hot(train_batch_graph.y, depth=num_classes)
        )

        losses = tf.reduce_mean(losses)

    vars = tape.watched_variables()
    grads = tape.gradient(losses, vars)
    optimizer.apply_gradients(zip(grads, vars))

    if step % 20 == 0:
        
        accuracy = gevaluate()
        print("step = {}\tloss = {}\taccuracy = {}".format(step, losses, accuracy))

for step in range(20000):#loss = 0.1952894926071167 accuracy = 0.7761557102203369
    train_batch_graph = next(train_batch_generator)
    
    with tf.GradientTape() as tape:
        logits = forward(train_batch_graph, training=True)
        losses = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits,
            labels=tf.one_hot(train_batch_graph.y, depth=num_classes)
        )

        losses = tf.reduce_mean(losses)

    vars = tape.watched_variables()
    grads = tape.gradient(losses, vars)
    optimizer.apply_gradients(zip(grads, vars))

    if step % 20 == 0:
        mean_loss = tf.reduce_mean(losses)
        accuracy = evaluate()
        print("step = {}\tloss = {}\taccuracy = {}".format(step, losses, accuracy))