# coding=utf-8
import os
#利用图节点互注意力实现图节点分类,与图卷积（GCN)相似，它也采用邻居聚合的方法（包括自环）更新特征
#只不过在聚合过程中，若邻居与中心节点的特征互注意力（包括与自己的自注意力）得分高，该邻居的聚合权重也会更高
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tf_geometric as tfg
import tensorflow as tf
from tensorflow import keras


graph, (train_index, valid_index, test_index) = tfg.datasets.CoraDataset().load_data()

num_classes = graph.y.max() + 1
drop_rate = 0.2

class GAT(tf.keras.Model):#多头图注意力

    def __init__(self, rate, num_classes):
        super(GAT, self).__init__()
        self.rate = rate
        self.num_classes = num_classes
        self.gat0 = tfg.layers.GAT(64, activation=tf.nn.relu, num_heads=8, 
                                drop_rate=self.rate, attention_units=8)#attention_units * num_heads = 维度（64）
        self.gat1 = tfg.layers.GAT(self.num_classes, drop_rate=0.6, attention_units=1)
        self.dropout = keras.layers.Dropout(self.rate)

    def call(self, graph):
        h = graph.x
        h = self.dropout(h)
        h = self.gat0([h, graph.edge_index, graph.edge_weight], cache=graph.cache)#GAT操作格式
        h = self.dropout(h)
        h = self.gat1([h, graph.edge_index, graph.edge_weight], cache=graph.cache)
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
gat = GAT(drop_rate, num_classes)

def train_step():
    for step in range(2000):
        with tf.GradientTape() as tape:
            logits = gat.call(graph)
            loss = compute_loss(logits, train_index, tape.watched_variables())

        vars = gat.trainable_variables
        grads = tape.gradient(loss, vars)
        optimizer.apply_gradients(zip(grads, vars))

        if step % 20 == 0:
            accuracy = evaluate(logits)
            print("step = {}\tloss = {}\taccuracy = {}".format(step, loss, accuracy))

train_step()