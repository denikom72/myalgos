# Bubble Sort
def bubble_sort(arr):
    """
    Use bubble sort when simplicity is more important than efficiency.
    Not recommended for large datasets due to its O(n^2) time complexity.
    """
    n = len(arr)
    for i in range(n - 1):
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

# Quick Sort
def quick_sort(arr):
    """
    Use quick sort for efficient sorting of large datasets.
    Provides an average-case time complexity of O(n log n).
    """
    if len(arr) <= 1:
        return arr
    pivot = arr[0]
    left = [x for x in arr[1:] if x < pivot]
    right = [x for x in arr[1:] if x >= pivot]
    return quick_sort(left) + [pivot] + quick_sort(right)

# Merge Sort
def merge_sort(arr):
    """
    Use merge sort when you need a stable, efficient sorting algorithm.
    Provides consistent O(n log n) time complexity.
    """
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Binary Search
def binary_search(arr, target):
    """
    Use binary search to efficiently search for an element in a sorted array.
    Has a time complexity of O(log n).
    """
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            return mid
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# Depth-First Search (DFS)
class Graph:
    def __init__(self, vertices):
        self.vertices = vertices
        self.adj = {}
        for vertex in range(vertices):
            self.adj[vertex] = []

    def add_edge(self, v, w):
        self.adj[v].append(w)

    def dfs(self, start):
        """
        Use depth-first search to traverse and explore all vertices in a graph.
        Often used for pathfinding and graph analysis.
        """
        visited = set()
        result = []
        self._dfs_util(start, visited, result)
        return result

    def _dfs_util(self, v, visited, result):
        visited.add(v)
        result.append(v)
        for neighbor in self.adj[v]:
            if neighbor not in visited:
                self._dfs_util(neighbor, visited, result)

# Breadth-First Search (BFS)
class Graph:
    def __init__(self, vertices):
        self.vertices = vertices
        self.adj = {}
        for vertex in range(vertices):
            self.adj[vertex] = []

    def add_edge(self, v, w):
        self.adj[v].append(w)

    def bfs(self, start):
        """
        Use breadth-first search to find the shortest path in an unweighted graph.
        Also used in web crawling and network routing.
        """
        visited = set()
        result = []
        queue = [start]
        visited.add(start)
        while queue:
            vertex = queue.pop(0)
            result.append(vertex)
            for neighbor in self.adj[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return result

# Dijkstra's Algorithm
class Graph:
    def __init(self, vertices):
        self.vertices = vertices
        self.graph = {}
        for vertex in range(vertices):
            self.graph[vertex] = {}

    def add_edge(self, u, v, weight):
        self.graph[u][v] = weight

    def dijkstra(self, start):
        """
        Use Dijkstra's algorithm to find the shortest path in a weighted graph with non-negative edges.
        Helpful in navigation and network routing.
        """
        dist = {vertex: float('inf') for vertex in range(self.vertices)}
        dist[start] = 0
        visited = set()
        for _ in range(self.vertices - 1):
            u = self._min_distance(dist, visited)
            visited.add(u)
            for v, weight in self.graph[u].items():
                if v not in visited and dist[u] + weight < dist[v]:
                    dist[v] = dist[u] + weight
        return dist

    def _min_distance(self, dist, visited):
        min_distance = float('inf')
        min_vertex = -1
        for vertex in range(self.vertices):
            if dist[vertex] < min_distance and vertex not in visited:
                min_distance = dist[vertex]
                min_vertex = vertex
        return min_vertex

# Floyd-Warshall Algorithm
def floyd_warshall(graph):
    """
    Use the Floyd-Warshall algorithm for finding all shortest paths in a weighted graph.
    Suitable for small graphs or when negative edge weights are allowed.
    """
    n = len(graph)
    dist = [list(row) for row in graph]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist

# Kruskal's Algorithm
class Graph:
    def __init__(self, vertices):
        self.vertices = vertices
        self.edges = []

    def add_edge(self, u, v, weight):
        self.edges.append((u, v, weight))

    def kruskal_mst(self):
        """
        Use Kruskal's algorithm to find the minimum spanning tree in a weighted graph.
        Often used in network design and clustering.
        """
        result = []
        self.edges.sort(key=lambda edge: edge[2])
        parent = list(range(self.vertices))
        for u, v, weight in self.edges:
            if self._is_cycle(parent, u, v):
                continue
            result.append((u, v, weight))
            self._union(parent, u, v)
        return result

    def _find(self, parent, i):
        if parent[i] == i:
            return i
        return self._find(parent, parent[i])

    def _union(self, parent, x, y):
        root_x = self._find(parent, x)
        root_y = self._find(parent, y)
        parent[root_x] = root_y

    def _is_cycle(self, parent, x, y):
        root_x = self._find(parent, x)
        root_y = self._find(parent, y)
        return root_x == root_y

# Topological Sort
def topological_sort(graph):
    """
    Use topological sort for ordering tasks with dependencies.
    Commonly applied in scheduling and build systems.
    """
    result = []
    visited = set()
    stack = []

    def dfs(node):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)
        stack.append(node)

    for node in graph:
        if node not in visited:
            dfs(node)

    while stack:
        result.append(stack.pop())
    return result

# Knapsack Problem (Dynamic Programming)
def knapsack_problem(values, weights, capacity):
    """
    Use dynamic programming to solve the knapsack problem and find the most valuable combination of items within a weight limit.
    Applied in resource allocation and optimization.
    """
    n = len(values)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    for i in range(n + 1):
        for w in range(capacity + 1):
            if i == 0 or w == 0:
                dp[i][w] = 0
            elif weights[i - 1] <= w:
                dp[i][w] = max(values[i - 1] + dp[i - 1][w - weights[i - 1]], dp[i - 1][w])
            else:
                dp[i][w] = dp[i - 1][w]
    result = []
    i, w = n, capacity
    while i > 0 and w > 0:
        if dp[i][w] != dp[i - 1][w]:
            result.append(i - 1)
            w -= weights[i - 1]
        i -= 1
    return result

# Longest Common Subsequence (Dynamic Programming)
def longest_common_subsequence(X, Y):
    """
    Use dynamic programming to find the longest common subsequence between two sequences.
    Commonly used in text comparison and DNA analysis.
    """
    m = len(X)
    n = len(Y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if X[i - 1] == Y[j - 1]:
            lcs.append(X[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return lcs[::-1]

# You can test the algorithms here

# Bubble Sort
arr = [64, 34, 25, 12, 22, 11, 90]
bubble_sort(arr)
print("Bubble Sort Result:", arr)

# Quick Sort
arr = [64, 34, 25, 12, 22, 11, 90]
quick_sort(arr)
print("Quick Sort Result:", arr)

# Merge Sort
arr = [64, 34, 25, 12, 22, 11, 90]
merge_sort(arr)
print("Merge Sort Result:", arr)

# Binary Search
arr = [11, 12, 22, 25, 34, 64, 90]
target = 25
index = binary_search(arr, target)
print(f"Binary Search Result: Element {target} found at index {index}")

# Depth-First Search (DFS)
g = Graph(4)
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 2)
g.add_edge(2, 0)
g.add_edge(2, 3)
g.add_edge(3, 3)
dfs_result = g.dfs(2)
print("DFS Result:", dfs_result)

# Breadth-First Search (BFS)
g = Graph(4)
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 2)
g.add_edge(2, 0)
g.add_edge(2, 3)
g.add_edge(3, 3)
bfs_result = g.bfs(2)
print("BFS Result:", bfs_result)

# Dijkstra's Algorithm
g = Graph(5)
g.add_edge(0, 1, 2)
g.add_edge(0, 2, 4)
g.add_edge(1, 2, 1)
g.add_edge(1, 3, 7)
g.add_edge(2, 3, 4)
dijkstra_result = g.dijkstra(0)
print("Dijkstra's Algorithm Result:", dijkstra_result)

# Floyd-Warshall Algorithm
graph = [
    [0, 5, float('inf'), 10],
    [float('inf'), 0, 3, float('inf')],
    [float('inf'), float('inf'), 0, 1],
    [float('inf'), float('inf'), float('inf'), 0]
]
floyd_warshall_result = floyd_warshall(graph)
print("Floyd-Warshall Algorithm Result:")
for row in floyd_warshall_result:
    print(row)

# Kruskal's Algorithm
g = Graph(4)
g.add_edge(0, 1, 10)
g.add_edge(0, 2, 6)
g.add_edge(0, 3, 5)
g.add_edge(1, 3, 15)
g.add_edge(2, 3, 4)
kruskal_result = g.kruskal_mst()
print("Kruskal's Algorithm Result:", kruskal_result)

# Topological Sort
graph = {
    0: [1, 2],
    1: [3],
    2: [3],
    3: []
}
topological_sort_result = topological_sort(graph)
print("Topological Sort Result:", topological_sort_result)

# Knapsack Problem
values = [60, 100, 120]
weights = [10, 20, 30]
capacity = 50
knapsack_result = knapsack_problem(values, weights, capacity)
print("Knapsack Problem Result:", knapsack_result)

# Longest Common Subsequence
X = "AGGTAB"
Y = "GXTXAYB"
lcs_result = longest_common_subsequence(X, Y)
print("Longest Common Subsequence Result:", lcs_result)
