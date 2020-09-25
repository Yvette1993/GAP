import random
import torch

def gene_random_graph_data(vertices, edges, file_path="random_data.txt"):
    max_edge = vertices * (vertices -1) / 2
    if max_edge < edges:
        raise ValueError("edges is overflow the max_edge for vertices:%d"%(vertices))
    edge_set = set()
    while len(edge_set) < edges*2:
        node1 = random.randint(0, vertices-1)
        node2 = random.randint(0, vertices -1)
        while node1 == node2:
            node2 = random.randint(0, vertices-1)
        edge_set.add((node1, node2))
        edge_set.add((node2, node1))
    with open(file_path, "w") as f:
        f.write("vertices: %d\n"%(vertices))
        for (i,j) in edge_set:
            f.write(str(i)+" " +str(j)+"\n")

def load_data_file(file_path="random_data.txt"):
    with open(file_path, "r") as f:
        vertices = f.readline()
        vertices.strip()
        vertices = int(vertices.split(" ")[1])
        A = torch.zeros((vertices, vertices))
        for edge in f:
            temp = edge.split(" ")
            A[int(temp[0])][int(temp[1])] = 1
        return A

if __name__ == "__main__":
    gene_random_graph_data(500, 9000)
    # A = load_data_file()

