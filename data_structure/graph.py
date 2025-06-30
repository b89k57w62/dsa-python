from collections import deque


class GraphAdjMat:
    def __init__(self, vertices: list[int], edges: list[list[int]]):
        self.vertices = []
        self.adj_mat = []

        for val in vertices:
            self.add_vertex(val)

        for edge in edges:
            self.add_edge(edge[0], edge[1])

    def size(self):
        return len(self.vertices)

    def add_vertex(self, val: int):
        self.vertices.append(val)
        new_row = [0] * self.size()
        self.adj_mat.append(new_row)
        for row in self.adj_mat:
            row.append(0)

    def remove_vertex(self, index):
        if index >= self.size():
            raise ValueError("Invalid index")
        self.vertices.pop(index)
        self.adj_mat.pop(index)
        for row in self.adj_mat:
            row.pop(index)

    def add_edge(self, i, j):
        if i < 0 or i >= self.size() or j < 0 or j >= self.size() or i == j:
            raise ValueError("Invalid edge")
        self.adj_mat[i][j] = 1
        self.adj_mat[j][i] = 1

    def remove_edge(self, i, j):
        if i < 0 or i >= self.size() or j < 0 or j >= self.size() or i == j:
            raise ValueError("Invalid edge")
        self.adj_mat[i][j] = 0
        self.adj_mat[j][i] = 0


def graph_bfs(graph: GraphAdjMat, start: int):
    res = []
    visited = set()
    queue = deque()
    queue.append(start)
    while len(queue) > 0:
        vet = queue.popleft()
        res.append(vet)
        for adj_vet in graph.adj_mat[vet]:
            if adj_vet not in visited:
                queue.append(adj_vet)
                visited.add(adj_vet)
    return res


def dfs_helper(graph: GraphAdjMat, vet, visited, res):
    res.append(vet)
    visited.add(vet)
    for adj_vet in graph.adj_mat[vet]:
        if adj_vet in visited:
            continue
        dfs_helper(graph, adj_vet, visited, res)


def graph_dfs(graph: GraphAdjMat, start: int):
    res = []
    visited = set()
    dfs_helper(graph, start, visited, res)
    return res
