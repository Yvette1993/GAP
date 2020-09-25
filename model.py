import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Cut_Blance_loss(torch.nn.Module):
    def __init__(self,adj):
        super(Cut_Blance_loss, self).__init__()
        self.adj = adj
    def forward(self,Y):
        A = self.adj
        n = Y.shape[0]
        g = Y.shape[1]
        D = torch.sum(A, dim=1)
        Gamma = torch.mm(Y.t(), D.unsqueeze(1).float())
        cut_loss = 0
        for i in range(g):
            e_cut = torch.mm(Y[:,i].unsqueeze(1), (1-Y[:,i]).unsqueeze(0))
            e_cut = e_cut * A
            n_cut = e_cut.sum() / Gamma[i,0]
            cut_loss += n_cut
        balance_loss = torch.pow((torch.sum(Y, dim=0) - n/g), 2)
        balance_loss = torch.sum(balance_loss)
        loss = cut_loss + balance_loss
        return loss


class GAP(torch.nn.Module):
    def __init__(self, number_class, input_data, embed_dim, linear_dim=128):
        super(GAP,self).__init__()
        self.num_class = number_class
        self.input_data = input_data
        self.graphsage = GraphSageFeature(
            input_data, embed_dim, input_data.shape[1], embed_dim, aggr_method="mean")
        self.liner1 = nn.Linear(embed_dim, linear_dim)
        self.liner2 = nn.Linear(linear_dim, number_class)
        self.softmax = nn.Softmax(dim=1)
        nn.init.xavier_normal_(self.liner1.weight)
        nn.init.xavier_normal_(self.liner2.weight)

    def forward(self, nodes, adj, sample=10):
        x = self.graphsage(nodes, adj, sample)
        x = self.liner1(x)
        x = self.liner2(x)
        x = self.softmax(x)
        return x


class GraphSageFeature(torch.nn.Module):
    def __init__(self, data, embed_dim, input_feature_dim, output_feature_dim, aggr_method="max"):
        super(GraphSageFeature, self).__init__()
        self.h0 = H0(data)
        self.aggr1 = Aggrator(self.h0, input_feature_dim,
                              input_feature_dim, aggr_method)
        self.en1 = Encoder(self.h0, self.aggr1, input_feature_dim, embed_dim)
        self.aggr2 = Aggrator(self.en1, embed_dim,
                              embed_dim, aggr_method)
        self.en2 = Encoder(self.en1, self.aggr2, embed_dim, output_feature_dim)

    def forward(self, nodes, adj, sample):
        out = self.en2(nodes, adj, sample)
        return out


class Aggrator(torch.nn.Module):
    def __init__(self, features, in_dim, out_dim, aggr_method="max"):
        super(Aggrator, self).__init__()
        self.aggr_method = aggr_method
        self.features = features
        self.in_dim = in_dim
        self.out_dim = out_dim
        if aggr_method == "max":
            # self.weight = torch.nn.Parameter(
            #     torch.FloatTensor(feature_dim, embed_dim))
            # self.bias = torch.nn.Parameter(torch.FloatTensor(embed_dim))
            self.linear = torch.nn.Linear(in_dim,out_dim)

    def forward(self, nodes_index, adj, number_samp=10):
        nodes_neigh = []
        for index in nodes_index:
            node_neigh = torch.nonzero(adj[index], as_tuple=False)
            numb = len(node_neigh)
            if numb >= number_samp:
                rand_index = torch.randperm(numb)[:number_samp]
                node_neigh = node_neigh[rand_index, 0]
                nodes_neigh.append(node_neigh)
            else:
                nodes_neigh.append(node_neigh[:, 0])
        nodes_neigh = torch.unique(torch.cat(nodes_neigh))
        embed_matrix = self.features(nodes_neigh, adj, number_samp)
        temp = adj[nodes_index]
        mask = temp.t()[nodes_neigh].t()
        if self.aggr_method == "max":
            embed_matrix = self.linear(embed_matrix)
            out = torch.zeros((len(nodes_index),len(nodes_neigh),self.in_dim)).cuda()
            for index in range(len(nodes_index)):
                neigh_index = torch.nonzero(mask[index],as_tuple=False)
                out[index][neigh_index][:] = embed_matrix[neigh_index][:]
            out = F.relu(out)
            out,_ = torch.max(out,dim=1)
        elif self.aggr_method == "mean":
            
            # mask = torch.index_select(temp, 1, nodes_neigh)
            
            # mask = mask.float()
            # mask = mask.div(mask.sum(1, keepdim=True))
            out = mask.mm(embed_matrix)
            out = out.div(mask.sum(1,keepdim=True))
            # out = mask.mm(embed_matrix)
        return out


class Encoder(torch.nn.Module):
    def __init__(self, features, aggregator, in_dim, out_dim):
        super(Encoder, self).__init__()
        self.features = features
        self.aggrator = aggregator
        self.in_dim = in_dim
        self.out_dim = out_dim
        # self.weight = nn.Parameter(torch.FloatTensor(feature_dim, embed_dim*2))
        # self.bias = nn.Parameter(torch.FloatTensor(feature_dim))
        self.linear = nn.Linear(in_dim*2,out_dim)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, nodes, adj, sample=10):
        neigh_features = self.aggrator(nodes, adj, sample)
        self_features = self.features(nodes, adj, sample)

        combined = torch.cat([self_features, neigh_features], dim=1)
        # out = F.relu(combined.mm(self.weight))
        # out = F.relu(self.weight.mm(combined.t()))
        out = self.linear(combined)
        out = F.relu(out)
        out = F.normalize(out)
        return out


class H0(torch.nn.Module):
    def __init__(self, data):
        super(H0, self).__init__()
        self.data = data
        self.embedding_layer = torch.nn.Embedding(*self.data.shape)
        self.embedding_layer.weight = nn.Parameter(data, requires_grad=False)

    def forward(self, nodes, adj=None, number_samp=None):
        return self.embedding_layer(nodes)


class SupervisedGraphSage(nn.Module):
    def __init__(self, number_class, graphsage, input_feature):
        super(SupervisedGraphSage, self).__init__()
        self.graphsage = graphsage
        self.linear = nn.Linear(input_feature, number_class, bias=False)

    def forward(self, nodes, adj, sample=10):
        embeds = self.graphsage(nodes, adj, sample)
        out = self.linear(embeds)
        return out


if __name__ == "__main__":
    data = torch.randn((80, 39))
    # feature = torch.nn.Embedding(80, 39)
    # feature.weight = nn.Parameter(data, requires_grad=False)
    model = GraphSageFeature(data, 64, 39, 239, "mean")
    adj = torch.randint(0, 2, (80, 80))
    for i in range(80):
        adj[i, i] = 0
        for j in range(i):
            adj[i, j] = adj[j, i]
    rand_indices = np.random.permutation(80)
    batch_nodes_list = rand_indices[:10]
    batch_nodes = torch.LongTensor(batch_nodes_list)
    model(batch_nodes, adj, 5)
