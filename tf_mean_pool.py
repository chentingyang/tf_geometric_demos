#利用gcn作为编码器，再用mean_pool将图聚合成一个超级节点，以实现图分类

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import tf_geometric as tfg
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# TU Datasets: https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
# COLLAB is a large dataset, which may costs 5 minutes for processing.
# tfg will automatically cache the processing result after the first processing.
# Thus, you can load it with only few seconds then.
graph_dicts = tfg.datasets.TUDataset("NCI1").load_data()

# Since a TU dataset may contain node_labels, node_attributes etc., each of which can be used as node features
# We process each graph as a dict and return a list of dict for graphs
# You can easily construct you Graph object with the data dict

num_node_labels = np.max([np.max(graph_dict["node_labels"]) for graph_dict in graph_dicts]) + 1#节点类别数, 37



def construct_graph(graph_dict):#构建图，将节点类别的one_hot作为节点特征(num_nodes, 37)，将图的类别标签作为图的ground_truth(1,)
    return tfg.Graph(
        x=keras.utils.to_categorical(graph_dict["node_labels"], num_node_labels),
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

class MeanPoolNetwork(keras.Model):
    def __init__(self, num_gins, units, num_classes, *args, **kwargs):
        """
        Demo GIN based Pooling Model
        :param num_gins: number of GIN layers
        :param units: Positive integer, dimensionality of the each GIN layer.
        :param num_classes: number of classes (for graph classification)
        """
        super().__init__(*args, **kwargs)

        self.gcn0 = tfg.layers.GCN(64, activation=tf.nn.relu)
        self.gcn1 = tfg.layers.GCN(32, activation=tf.nn.relu)
        self.drop1 = keras.layers.Dropout(0.3)
        self.mlp = keras.Sequential([
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(num_classes)
        ])#输出层

    def call(self, batch_graph, training=False, mask=None):

        h = batch_graph.x#(batch中节点的总个数，37)
        edge_index = batch_graph.edge_index#(2, batch中边的总个数)
        edge_weight = batch_graph.edge_weight
        node_graph_index = batch_graph.node_graph_index#(节点总个数)，利用它将batch输入拆分成子图

        h  = self.gcn0([h, edge_index, edge_weight])#(num_nodes, 32)
        
        h = self.drop1(h)

        h = self.gcn1([h, edge_index, edge_weight])#图卷积编码器，图分类不需要指定图cache

        h = tfg.nn.sum_pool(h, node_graph_index)#sum_pool读出层，利用sum_pool维度聚合对各子图的节点归一化成一个超级节点，（num_graph,32)
        
        logits = self.mlp(h, training=training)#输出层,(num_graph, num_classes)
       
        return logits


model = MeanPoolNetwork(5, 32, num_classes)
batch_size = 512


def evaluate(graphs, batch_size):#在batch上进行评估
    accuracy_m = keras.metrics.Accuracy()

    for batch_graph in create_graph_generator(test_graphs, batch_size, shuffle=False, infinite=False):
        
        logits = model(batch_graph)
        preds = tf.argmax(logits, axis=-1)
        accuracy_m.update_state(batch_graph.y, preds)#每个batch更新一次精度

    return accuracy_m.result().numpy()


# optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
train_batch_generator = create_graph_generator(train_graphs, batch_size, shuffle=True, infinite=True)


best_test_acc = 0
for step in range(0, 1000):#loss = 0.5063184499740601    train_acc = 0.7055960893630981  test_acc=0.7445255517959595
    batch_graph = next(train_batch_generator)
    
    with tf.GradientTape() as tape:
        
        logits = model(batch_graph, training=True)
        losses = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits,
            labels=tf.one_hot(batch_graph.y, depth=num_classes)
        )

        loss = tf.reduce_mean(losses)

    vars = tape.watched_variables()
    grads = tape.gradient(loss, vars)
    optimizer.apply_gradients(zip(grads, vars))

    if step % 10 == 0:
        
        train_acc = evaluate(train_graphs, batch_size)
        test_acc = evaluate(test_graphs, batch_size)

        if best_test_acc < test_acc:
            best_test_acc = test_acc#更改最佳测试精度

        if test_acc < best_test_acc * 0.85:
            break

        print("step = {}\tloss = {}\ttrain_acc = {}\ttest_acc={}".format(step, loss, train_acc, best_test_acc))