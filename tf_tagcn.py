# coding=utf-8
import os
#利用tagcn实现图节点分类,它在聚合节点邻居时，不止聚合节点的一阶邻居（等同于图卷积GCN)，还聚合节点的2,3，...k阶邻居（自环renorm默认为False）
#也就是说，一个k阶的TAGCN相当于一个同时包括了[一层不包括全连层的GCN,2层不包括全连层的GCN,...,k层不包括全连层的GCN]的输出的列表，
#它把这些输出前后拼接起来，最后经过全连层进行维度变换
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tf_geometric as tfg
import tensorflow as tf
from tensorflow import keras


graph, (train_index, valid_index, test_index) = tfg.datasets.CoraDataset().load_data()

num_classes = graph.y.max() + 1
drop_rate = 0.2

class TAGCN(tf.keras.Model):#

    def __init__(self, rate, num_classes):
        super(TAGCN, self).__init__()
        self.rate = rate
        self.num_classes = num_classes
        self.tagcn0 = tfg.layers.TAGCN(64, K=3, activation=tf.nn.relu, renorm=True) 
        self.tagcn1 = tfg.layers.TAGCN(self.num_classes, K=3, renorm=True)
        self.dropout = keras.layers.Dropout(self.rate)

    def call(self, graph):
        h = graph.x
        h = self.dropout(h)
        h = self.tagcn0([h, graph.edge_index, graph.edge_weight], cache=graph.cache)
        h = self.dropout(h)
        h = self.tagcn1([h, graph.edge_index, graph.edge_weight], cache=graph.cache)
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
tagcn = TAGCN(drop_rate, num_classes)

def train_step():
    for step in range(2000):
        with tf.GradientTape() as tape:
            logits = tagcn.call(graph)
            loss = compute_loss(logits, train_index, tape.watched_variables())

        vars = tagcn.trainable_variables
        grads = tape.gradient(loss, vars)
        optimizer.apply_gradients(zip(grads, vars))

        if step % 20 == 0:
            accuracy = evaluate(logits)
            print("step = {}\tloss = {}\taccuracy = {}".format(step, loss, accuracy))

train_step()