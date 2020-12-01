#GIN模型在图节点邻居特征的每一跳聚合操作之后，又与自身的原始特征混合起来。并在最后使用可以拟合任意规则的全连接网络进行处理，
#即先节点邻居聚合（不包括自环），再加入自身特征，再通过全连层，使其具有单射特性(即不同节点间结构对应的聚合后节点维度embedding是不同的）。
#在特征混合的过程中，引入了一个可学习参数对自身特征进行调节，并将调节后的特征与聚合后的邻居特征进行相加（不包括自环），再经过全连层.
#这与图卷积（单个全连层）不同之处在于自身特征经过参数调节，而且特征聚合后要经过多个全连层，使其单射特性得到进一步加强，这是一个两层的感知机（MLP)
#通过GIN学习的节点embeddings可以用于类似于节点分类、连接预测这样的任务。
#对于图分类任务，利用gin作为编码器，可以进一步添加读出层，达到：给定独立的节点的embeddings，生成整个图的embedding


import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import tf_geometric as tfg
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split


# TU Datasets: https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
# COLLAB is a large dataset, which may costs 5 minutes for processing.
# tfg will automatically cache the processing result after the first processing.
# Thus, you can load it with only few seconds then.
graph_dicts = tfg.datasets.TUDataset("NCI1").load_data()

# Since a TU dataset may contain node_labels, node_attributes etc., each of which can be used as node features
# We process each graph as a dict and return a list of dict for graphs
# You can easily construct you Graph object with the data dict

num_node_labels = np.max([np.max(graph_dict["node_labels"]) for graph_dict in graph_dicts]) + 1#节点类别数, 37


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

class GINPoolNetwork(keras.Model):
    def __init__(self, num_gins, units, num_classes, *args, **kwargs):
        """
        Demo GIN based Pooling Model
        :param num_gins: number of GIN layers
        :param units: Positive integer, dimensionality of the each GIN layer.
        :param num_classes: number of classes (for graph classification)
        """
        super().__init__(*args, **kwargs)

        self.gins = [
            tfg.layers.GIN(
                keras.Sequential([
                    keras.layers.Dense(units, activation=tf.nn.relu),
                    keras.layers.Dense(units),
                    keras.layers.BatchNormalization(),
                    keras.layers.Activation(tf.nn.relu)
                ]),eps=.5#GIN结构，节点自身特征先进行权重调节，再将邻居特征进行聚合（不包括自环），与节点特征相加和，最后再经过2个全连层
                #其数学公式为H(c, X) = g((1+eps)h(c) + f(X)) 其中g为多层全连层，内部前半部分是节点自身的权重调节（可训练），后半部分是邻居聚合操作（包括自环）
            )
            for _ in range(num_gins)  # num_gins blocks
        ]#GIN编码器

        self.mlp = keras.Sequential([
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(num_classes)
        ])#输出层

    def call(self, batch_graph, training=False, mask=None):

        h = batch_graph.x#(batch中节点的总个数，37)
        edge_index = batch_graph.edge_index#(2, batch中边的总个数)
        edge_weight = batch_graph.edge_weight
        node_graph_index = batch_graph.node_graph_index#(节点总个数)，标识节点属于哪张图的索引，利用它将batch输入拆分成子图

        hidden_outputs = []
        

        for gin in self.gins:
            h = gin([h, edge_index, edge_weight], training=training)#图分类不需要指定图cache
            hidden_outputs.append(h)

        h = tf.concat(hidden_outputs, axis=-1)#(num_nodes, units*num_gins),对四层gin层的输出进行拼接
        h = tfg.nn.sum_pool(h, node_graph_index)#sum_pool读出层，利用sum_pool维度聚合对各子图的节点归一化成一个超级节点，（num_graph,units*num_gins)
        logits = self.mlp(h, training=training)#输出层
        return logits


model = GINPoolNetwork(5, 64, num_classes)
batch_size = 512


def evaluate(graphs, batch_size):
    accuracy_m = keras.metrics.Accuracy()

    for batch_graph in create_graph_generator(test_graphs, batch_size, shuffle=False, infinite=False):
        logits = model(batch_graph)
        preds = tf.argmax(logits, axis=-1)
        accuracy_m.update_state(batch_graph.y, preds)

    return accuracy_m.result().numpy()


# optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
train_batch_generator = create_graph_generator(train_graphs, batch_size, shuffle=True, infinite=True)


best_test_acc = 0
for step in range(0, 1000):#train_acc = 0.6872127652168274   test_acc=0.7007299065589905，效果不好
    batch_graph = next(train_batch_generator)#每次训练一个batch
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

        if test_acc < best_test_acc * 0.9:
            break

        print("step = {}\tloss = {}\ttrain_acc = {}\ttest_acc={}".format(step, loss, train_acc, test_acc))