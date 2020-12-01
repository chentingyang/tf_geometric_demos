#利用图自编码器实现对破损的图的重构或挖掘图节点间的深层联系
import tf_geometric as tfg
import tensorflow as tf
from tensorflow import keras
from tf_geometric.utils.graph_utils import edge_train_test_split, negative_sampling
import numpy as np
from sklearn.metrics import roc_auc_score

graph, (train_index, valid_index, test_index) = tfg.datasets.CoraDataset().load_data()

#graph.edge_index.shape=(2, 10556)
#任务是通过自编码器对节点的特征维度进行编码，进而挖掘节点之间的相互关系，从而对节点间边的链接关系进行预测
#首先要将边进行分离，分离成训练集和测试集， 同时转为无向边
undirected_train_edge_index, undirected_test_edge_index, _, _ = edge_train_test_split(
    edge_index=graph.edge_index,
    num_nodes=graph.num_nodes,
    mode='undirected', 
    test_size=0.15
)
print(undirected_train_edge_index.shape, undirected_test_edge_index.shape)#(2, 4486) (2, 792)

# 通过replace=False的负采样，来为测试集产生负样本（不存在的边），以满足验证评估要求
undirected_test_neg_edge_index = negative_sampling(
    num_samples=undirected_test_edge_index.shape[1],
    num_nodes=graph.num_nodes,
    edge_index=graph.edge_index,
    replace=False#不生成重复的负采样样本
)#(2, 792)

train_graph = tfg.Graph(x=graph.x, edge_index=undirected_train_edge_index).convert_edge_to_directed()
t_graph = tfg.Graph(x=graph.x, edge_index=undirected_train_edge_index)
#将训练图重新变回有向边，也就是把每条无向边首尾颠倒再复制一次
#print(train_graph.edge_index.shape)， (2, 8972)训练图中有8972条边，是将测试集中的边从原图中拆除后得到的


#利用图卷积编码器进行边的恢复
class Encoder(tf.keras.Model):#原始的图自编码器使用图卷积网络（GCN），基于节点的特征和边，为节点学习高阶特征

	def __init__(self, rate, embedding_size):
		super(Encoder, self).__init__()
		self.rate = rate
		self.embedding_size = embedding_size#编码维度
		self.gcn0 = tfg.layers.GCN(32, activation=tf.nn.relu)
		self.gcn1 = tfg.layers.GCN(self.embedding_size, activation=tf.nn.relu)
		self.dropout = keras.layers.Dropout(self.rate)

	def call(self, graph):
		h = self.gcn0([graph.x, graph.edge_index, graph.edge_weight], cache=graph.cache)
		h = self.dropout(h)
		h = self.gcn1([h, graph.edge_index, graph.edge_weight], cache=graph.cache)
		return h


#对编码结果进行边的重构， 输入一对节点，若在图编码过程中挖掘得到二者间存在联系，则经过边重构后的输出结果应该为1（二者间存在边）
def predict_edge(embedded, edge_index):#embedded为encoder编码后的图， row,col为边的头尾节点索引
    row, col = edge_index
    embedded_row = tf.gather(embedded, row)#获得训练集中的边在编码后的图中的对应头节点的特征表示#（train_edge_index_size, embedding_dim)
    embedded_col = tf.gather(embedded, col)#获得训练集中的边在编码后的图中的对应尾节点的特征表示
    
    # 点乘
    logits = tf.reduce_sum(embedded_row * embedded_col, axis=-1)#（train_edge_index_size)
    
    return logits



def compute_loss(pos_edge_logits, neg_edge_logits):
	pos_losses = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=pos_edge_logits,
        labels=tf.ones_like(pos_edge_logits)#真样本标识为1
	)#先sigmoid再求交叉熵损失(logit和label的shape相同)

	neg_losses = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=neg_edge_logits,
        labels=tf.zeros_like(neg_edge_logits)#假样本标识为0
	)

	return tf.reduce_mean(pos_losses) + tf.reduce_mean(neg_losses)#对于测试集的真样本，输出值应为1，对于负采样的假样本，输出应为0


	


def evaluate(embedded):#评估(包括正样本和负样本)

    pos_edge_logits = predict_edge(embedded, undirected_test_edge_index)#(test_edge_index_size)
    neg_edge_logits = predict_edge(embedded, undirected_test_neg_edge_index)#(test_edge_index_size)

    pos_edge_scores = tf.nn.sigmoid(pos_edge_logits)
    neg_edge_scores = tf.nn.sigmoid(neg_edge_logits)

    y_true = tf.concat([tf.ones_like(pos_edge_scores), tf.zeros_like(neg_edge_scores)], axis=0)
    y_pred = tf.concat([pos_edge_scores, neg_edge_scores], axis=0)#正样本和负样本拼接

    auc_m = keras.metrics.AUC()
    auc_m.update_state(y_true, y_pred)

    return auc_m.result().numpy()#利用auc_score评估（评估回归预测的方法，类似分类的accuracy）


encoder1 = Encoder(0.2, 64)
encoder2 = Encoder(0.2, 64)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
ll = encoder2.call(t_graph)


def train_step():
	for step in range(1000):
		with tf.GradientTape() as tape:
			embedded = encoder1.call(train_graph)

			# 对训练样本进行负采样生成不存在的边
			train_neg_edge_index = negative_sampling(
				train_graph.num_edges,
				graph.num_nodes,
				edge_index=train_graph.edge_index
			)#不用replace=False是因为训练时可以有重复的负样本
			#对正负样本进行训练
			pos_edge_logits = predict_edge(embedded, train_graph.edge_index)
			neg_edge_logits = predict_edge(embedded, train_neg_edge_index)

			loss = compute_loss(pos_edge_logits, neg_edge_logits)

		vars = tape.watched_variables()
		grads = tape.gradient(loss, vars)
		optimizer.apply_gradients(zip(grads, vars))

		if step % 20 == 0:
			auc_score = evaluate(embedded)
			print("step = {}\tloss = {}\tauc_score = {}".format(step, loss, auc_score))


