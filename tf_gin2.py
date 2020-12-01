#利用图同构网络模型(GIN)作为encoder进行图节点分类,它可以实现图结构的单射，是复杂结构图上对图卷积的优化
# coding=utf-8
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tf_geometric as tfg
import tensorflow as tf
from tensorflow import keras
from tf_geometric.utils.graph_utils import add_self_loop_edge

graph, (train_index, valid_index, test_index) = tfg.datasets.CoraDataset().load_data()
print(graph.x.shape, graph.edge_index.shape)#(2708, 1433) (2, 10556)

num_classes = graph.y.max() + 1#7
drop_rate = 0.3
units = 64

class GIN(tf.keras.Model):

    def __init__(self, rate, units, num_classes):
        super(GIN, self).__init__()
        self.rate = rate
        self.num_classes = num_classes
        self.gin1 = tfg.layers.GIN(
                keras.Sequential([
                    keras.layers.Dense(units, activation=tf.nn.relu),
                    keras.layers.Dense(units),
                    keras.layers.BatchNormalization(),
                    keras.layers.Activation(tf.nn.relu)
                ]), eps=.5)#GIN结构，节点自身特征先进行权重调节，再将邻居特征进行聚合（不包括自环），与节点特征相加和，最后再经过2个全连层
                #其数学公式为G(x+1) = g((1+eps)G(x) + f(x)) 其中g为全连层，内部前半部分是节点自身的权重调节（可训练），后半部分是聚合操作（包括自环））
                ##gin结构中的第二个全连层不使用激活函数，而是先正则化后再使用激活函数
        self.gin2 = tfg.layers.GIN(
                keras.Sequential([
                    keras.layers.Dense(units, activation=tf.nn.relu),
                    keras.layers.Dense(self.num_classes),
                    keras.layers.BatchNormalization()
                ]), eps=.5)
        
        self.dropout = keras.layers.Dropout(self.rate)



    def call(self, graph):
        h = graph.x
        h = self.dropout(h)
        h = self.gin1([h, graph.edge_index, graph.edge_weight], cache=graph.cache)
        h = self.gin2([h, graph.edge_index, graph.edge_weight], cache=graph.cache)
        
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


optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
gin = GIN(drop_rate, units, num_classes)

def train_step():
    best_test_acc = 0#最佳精确度
    for step in range(500):#对于单张图训练次数不应太多
        with tf.GradientTape() as tape:
            logits = gin.call(graph)
            loss = compute_loss(logits, train_index, tape.watched_variables())

        vars = gin.trainable_variables
        grads = tape.gradient(loss, vars)
        optimizer.apply_gradients(zip(grads, vars))

        if step % 20 == 0:
            accuracy = evaluate(logits)
            print("step = {}\tloss = {}\taccuracy = {}".format(step, loss, accuracy))
            
            if accuracy > best_test_acc:#更新最佳精确度
                best_test_acc = accuracy
            
            if accuracy < (best_test_acc * 0.9):
                break
                

train_step()