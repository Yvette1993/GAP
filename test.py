from data import load_data_file
from torch_sparse import SparseTensor
import torch
from torch_geometric.nn import SAGEConv

adj = load_data_file()
adj_sparse = SparseTensor.from_torch_sparse_coo_tensor(adj)

x = torch.randn((5,10))
sage_conv = SAGEConv(10,9,True)
out = sage_conv(x,adj_sparse)
print(out)
print(type(out))

