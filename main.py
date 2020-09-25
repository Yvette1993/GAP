from model import GraphSageFeature, SupervisedGraphSage, GAP, Cut_Blance_loss
import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict
import torch
from data import load_data_file
from sklearn.decomposition import PCA


def load_cora():
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats), dtype=np.float32)
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("cora/cora.content") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            feat_data[i, :] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open("cora/cora.cites") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists


def run_cora():
    np.random.seed(1)
    random.seed(1)
    num_nodes = 2708
    feat_data, labels, adj_lists = load_cora()
    adj = torch.zeros((2708, 2708))
    for i, adjs in adj_lists.items():
        for j in adjs:
            adj[i, j] = 1
    adj = adj.cuda()
    feat_data = torch.from_numpy(feat_data)
    graphsage_feature = GraphSageFeature(feat_data, 128, 1433, 128, "max")
    super_graphase = SupervisedGraphSage(7, graphsage_feature, 128)

    super_graphase.cuda()
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])
    loss_func = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, super_graphase.parameters()), lr=0.7)
    times = []
    for batch in range(200):
        batch_nodes_list = train[:128]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        batch_nodes = torch.LongTensor(batch_nodes_list).cuda()
        output = super_graphase(batch_nodes, adj)
        loss = loss_func(output,
                         torch.LongTensor(labels[np.array(batch_nodes_list)]).squeeze().cuda())
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print(batch, loss.item())

    val_output = super_graphase(torch.LongTensor(val).cuda(), adj)
    print("Validation F1:", f1_score(
        labels[val][:, 0], val_output.data.cpu().numpy().argmax(axis=1), average="micro"))
    print("Average batch time:", np.mean(times))


def run_graph_cut():
    # adj = load_data_file()
    adj = np.fromfile("graph_matrix.bin").reshape(10,10)
    pca = PCA(1)
    data = pca.fit_transform(adj)
    # data = torch.ones((500, 1))
    data = torch.Tensor(data).cuda()
    
    adj = torch.Tensor(adj).cuda()
    # data = torch.sum(adj,dim=1)/ torch.sum(adj)
    # data = data.unsqueeze(1)
    # new_adj = torch.where(adj > 0, torch.Tensor([0.7]).cuda(), torch.Tensor([-0.3]).cuda())
    # new_adj.fill_diagonal_(0)
    # data = new_adj.mm(data)
    model = GAP(10, data, 128)
    model.cuda()
    rand_indices = np.random.permutation(10)
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    loss_func = Cut_Blance_loss(adj)
    loss_func.cuda()
    times = []
    for batch in range(200):
        batch_nodes_list = rand_indices
        random.shuffle(rand_indices)
        start_time = time.time()
        optimizer.zero_grad()
        batch_nodes = torch.LongTensor(batch_nodes_list).cuda()
        output = model(batch_nodes, adj)
        loss = loss_func(output)
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print(batch, loss.item())
    print("Average batch time:", np.mean(times))
    torch.save(model.state_dict(), "simple_test.pth")
    output = model(batch_nodes,adj)
    output_label = output.data.cpu().numpy().argmax(axis=1)
    print(output_label)
    model_output = torch.zeros((10,10)).cuda()
    # my_output= [[1,0],[1,0],[0,1],[0,1],[1,0],[1,0],[1,0],[0,1],[0,1],[0,1]]
    my_output = torch.zeros((10,10)).cuda()
    # my_output = torch.Tensor(my_output).cuda()
    for i in range(10):
        model_output[i][output_label[i]] = 1
        my_output[i][i] = 1
    print("model_loss:", loss_func(model_output))
    print("my_loss:",loss_func(my_output))


def test_graph_cut():
    adj = load_data_file("test_data.txt")
    pca = PCA(256)
    data = pca.fit_transform(adj.numpy())
    data = torch.Tensor(data).cuda()
    adj = adj.cuda()
    model = GAP(4, data, 128)
    parameter = torch.load("simple_test.pth")
    model.load_state_dict(parameter)
    model.cuda()


if __name__ == "__main__":
    run_cora()
