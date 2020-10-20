from gcn.vertex import Vertex
import torch


class vGraph:
    def __init__(self, N, conn_data, reward_data):
        self.N = N
        self.vertexes = []
        for i in range(N):
            self.vertexes.append(Vertex(reward_data[i], conn_data[i]))

    def X(self):
        X = []
        for v in self.vertexes:
            X.append(v.features)
        X = torch.stack(X, 0)
        return X

    def adj(self, value_type='b'):
        if value_type != 'b' and value_type != 'w':
            return None
        adj = []
        for v in self.vertexes:
            adj.append(v.neighbors)
        adj = torch.stack(adj, 0)
        adj[adj == -1] = 0
        if value_type == 'b':
            adj[adj != 0] = 1
        i = torch.eye(self.N)
        adj = adj+i
        return adj

    def deg(self, waveA):
        deg = []
        for i in range(self.N):
            deg.append(torch.zeros(self.N))
            deg[i][i] = torch.sum(waveA[i])
        deg = torch.stack(deg, 0)
        return deg
