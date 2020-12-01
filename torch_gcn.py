import torch
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

#数据集加载
from torch_geometric.datasets import Planetoid
dataset = Planetoid(root='F:/py file/geometric', name='Cora')
print(len(dataset))#只有一张图
data = dataset[0]
print(data)#Data(edge_index=[2, 10556], test_mask=[2708], train_mask=[2708], val_mask=[2708], x=[2708, 1433], y=[2708])
#共有10556条边，2708个节点，每个节点有1433个特征
print(data.train_mask.sum().item())#data.train_mask是训练数据的mask标记,共有140条
print(data.x.shape)#(2708, 1433)
print(data.num_node_features)#特征维度：1433
print(dataset.num_classes)#类别数：7
print(data.edge_index)#(2, 10556)代表10556条边，分为两个数组，边从第一个数组中的元素指向第二个数组中相同位置的元素, 注意：在实际计算时每个节点还需要加上自身的环路

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 128)#第一个图卷积层
        self.conv2 = GCNConv(128, 16)#第二个图卷积层
        self.conv3 = GCNConv(16, dataset.num_classes)#第三个图卷积层

    def forward(self, data):
        x, edge_index = data.x, data.edge_index#节点和边

        x = self.conv1(x, edge_index)#(2708,128)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv2(x, edge_index)#(2708,16)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv3(x, edge_index)#(2708,7)
        

        return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model_dir = ('F:\\py file\\geometric\\model_dir\\init_model.pkl')
#网络训练
def train():
	model.train()
	for epoch in range(200):
	    optimizer.zero_grad()
	    out = model(data)
	    loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask])
	    loss.backward()
	    optimizer.step()
	    #保存模型
	    torch.save(model, model_dir)
	 
#测试
def eval():
	model = torch.load(model_dir)
	model.eval()
	_, pred = model(data).max(dim=1)#取输出维度上的最大值(argmax),shape=(2708)
	correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
	acc = correct / data.test_mask.sum().item()
	print('Accuracy: {:.4f}'.format(acc))

train()
eval()