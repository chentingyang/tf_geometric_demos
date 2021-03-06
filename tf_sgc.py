#利用简化的图卷积网络SGC(包括自环：renorm=True，默认为True)进行邻居聚合，实现节点分类
#SGC的邻居聚合的过程与GCN相同，它的简化点在于SGC认为与邻居聚合的过程相比，维度变换及其中的非线性激活并不重要
#与其他聚合网络每层进行一次邻居聚合，再进行一次维度变换不同，每层SGC先进行k次一阶邻居聚合（等同于k层不包括全连层的GCN),最后再经过全连层
#SGC是一个线性的处理过程
# coding=utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tf_geometric as tfg
import tensorflow as tf
from tensorflow import keras
from tf_geometric.datasets.cora import CoraDataset


graph, (train_index, valid_index, test_index) = CoraDataset().load_data()

num_classes = graph.y.max() + 1


model = tfg.layers.SGC(num_classes, k=2, renorm=True)#进行两次一阶邻居聚合（辐射到2阶邻居）


def forward(graph):
    h = model([graph.x, graph.edge_index, graph.edge_weight], cache=graph.cache)
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
    logits = forward(graph)
    logits = tf.nn.log_softmax(logits, axis=1)
    masked_logits = tf.gather(logits, mask)
    masked_labels = tf.gather(graph.y, mask)

    y_pred = tf.argmax(masked_logits, axis=-1, output_type=tf.int32)
    accuracy_m = keras.metrics.Accuracy()
    accuracy_m.update_state(masked_labels, y_pred)
    return accuracy_m.result().numpy()



optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)

for step in range(1,101):
    with tf.GradientTape() as tape:
        logits = forward(graph)
        logits = tf.nn.log_softmax(logits,axis=1)
        loss = compute_loss(logits, train_index, tape.watched_variables())

    vars = tape.watched_variables()
    grads = tape.gradient(loss, vars)
    optimizer.apply_gradients(zip(grads, vars))

    valid_acc = evaluate(valid_index)
    test_acc = evaluate(test_index)

    print("step = {}\tloss = {}\tvalid_acc = {}\ttest_acc = {}".format(step, loss, valid_acc, test_acc))