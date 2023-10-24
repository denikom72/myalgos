# Merge Sort
def merge_sort(arr)
  return arr if arr.length <= 1

  mid = arr.length / 2
  left = merge_sort(arr[0...mid])
  right = merge_sort(arr[mid..-1])
  merge(left, right)
end

def merge(left, right)
  result = []
  until left.empty? || right.empty?
    result << (left.first <= right.first ? left.shift : right.shift)
  end
  result.concat(left).concat(right)
end

# Binary Search
def binary_search(arr, target)
  left, right = 0, arr.length - 1
  while left <= right
    mid = left + (right - left) / 2
    if arr[mid] == target
      return mid
    elsif arr[mid] < target
      left = mid + 1
    else
      right = mid - 1
    end
  end
  -1
end

# Depth-First Search (DFS)
class Graph
  def initialize(vertices)
    @vertices = vertices
    @adj = {}
    (0...vertices).each { |vertex| @adj[vertex] = [] }
  end

  def add_edge(v, w)
    @adj[v] << w
  end

  def dfs(start)
    visited = Set.new
    result = []
    dfs_util(start, visited, result)
    result
  end

  private

  def dfs_util(v, visited, result)
    visited.add(v)
    result << v
    @adj[v].each do |neighbor|
      dfs_util(neighbor, visited, result) unless visited.include?(neighbor)
    end
  end
end

# Breadth-First Search (BFS)
class Graph
  def bfs(start)
    visited = Set.new
    result = []
    queue = [start]
    visited.add(start)

    until queue.empty?
      vertex = queue.shift
      result << vertex
      @adj[vertex].each do |neighbor|
        unless visited.include?(neighbor)
          visited.add(neighbor)
          queue << neighbor
        end
      end
    end

    result
  end
end

# Dijkstra's Algorithm
class Graph
  def dijkstra(start)
    dist = Array.new(@vertices, Float::INFINITY)
    dist[start] = 0
    visited = Set.new
    (0...@vertices - 1).each do |_|
      u = min_distance(dist, visited)
      visited.add(u)
      @adj[u].each do |v, weight|
        dist[v] = dist[v] + weight if !visited.include?(v) && dist[u] + weight < dist[v]
      end
    end
    dist
  end

  private

  def min_distance(dist, visited)
    min = Float::INFINITY
    min_index = -1
    dist.each_with_index do |val, idx|
      if !visited.include?(idx) && val <= min
        min = val
        min_index = idx
      end
    end
    min_index
  end
end

# Floyd-Warshall Algorithm
def floyd_warshall(graph)
  n = graph.length
  dist = Marshal.load(Marshal.dump(graph))
  (0...n).each do |k|
    (0...n).each do |i|
      (0...n).each do |j|
        if dist[i][j] > dist[i][k] + dist[k][j]
          dist[i][j] = dist[i][k] + dist[k][j]
        end
      end
    end
  end
  dist
end

# Kruskal's Algorithm
class Graph
  def kruskal_mst
    result = []
    @edges.sort_by! { |edge| edge[2] }
    parent = (0...@vertices).to_a

    @edges.each do |edge|
      u, v, weight = edge
      if find(parent, u) != find(parent, v)
        result << edge
        union(parent, u, v)
      end
    end

    result
  end

  private

  def find(parent, i)
    if parent[i] == i
      i
    else
      find(parent, parent[i])
    end
  end

  def union(parent, x, y)
    x_root = find(parent, x)
    y_root = find(parent, y)
    parent[x_root] = y_root
  end
end

# Topological Sort
def topological_sort(graph)
  result = []
  visited = Set.new
  stack = []

  dfs = lambda do |node|
    visited.add(node)
    graph[node].each do |neighbor|
      dfs.call(neighbor) unless visited.include?(neighbor)
    end
    stack.push(node)
  end

  graph.keys.each do |node|
    dfs.call(node unless visited.include?(node))
  end

  result.push(stack.pop) until stack.empty?
  result
end

# Knapsack Problem (Dynamic Programming)
def knapsack_problem(values, weights, capacity)
  n = values.length
  dp = Array.new(n + 1) { Array.new(capacity + 1, 0) }

  (0..n).each do |i|
    (0..capacity).each do |w|
      if i == 0 || w == 0
        dp[i][w] = 0
      elsif weights[i - 1] <= w
        dp[i][w] = [values[i - 1] + dp[i - 1][w - weights[i - 1]], dp[i - 1][w]].max
      else
        dp[i][w] = dp[i - 1][w]
      end
    end
  end

  result = []
  i, w = n, capacity
  while i.positive? && w.positive?
    if dp[i][w] != dp[i - 1][w]
      result.push(i - 1)
      w -= weights[i - 1]
    end
    i -= 1
  end

  result
end

# Longest Common Subsequence (Dynamic Programming)
def longest_common_subsequence(x, y)
  m = x.length
  n = y.length
  dp = Array.new(m + 1) { Array.new(n + 1, 0) }

  (1..m).each do |i|
    (1..n).each do |j|
      if x[i - 1] == y[j - 1]
        dp[i][j] = dp[i - 1][j - 1] + 1
      else
        dp[i][j] = [dp[i - 1][j], dp[i][j - 1]].max
      end
    end
  end

  lcs = []
  i, j = m, n
  while i.positive? && j.positive?
    if x[i - 1] == y[j - 1]
      lcs.push(x[i - 1])
      i -= 1
      j -= 1
    elsif dp[i - 1][j] > dp[i][j - 1]
      i -= 1
    else
      j -= 1
    end
  end

  lcs.reverse
end

# Test the algorithms

# Bubble Sort
arr = [64, 34, 25, 12, 22, 11, 90]
bubble_sort(arr)
puts "Bubble Sort Result: #{arr}"

# Quick Sort
arr = [64, 34, 25, 12, 22, 11, 90]
quick_sort(arr)
puts "Quick Sort Result: #{arr}"

# Merge Sort
arr = [64, 34, 25, 12, 22, 11, 90]
merge_sort(arr)
puts "Merge Sort Result: #{arr}"

# Binary Search
arr = [11, 12, 22, 25, 34, 64, 90]
target = 25
index = binary_search(arr, target)
puts "Binary Search Result: Element #{target} found at index #{index}"

# Depth-First Search (DFS)
g = Graph.new(4)
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 2)
g.add_edge(2, 0)
g.add_edge(2, 3)
g.add_edge(3, 3)
dfs_result = g.dfs(2)
puts "DFS Result: #{dfs_result}"

# Breadth-First Search (BFS)
g = Graph.new(4)
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 2)
g.add_edge(2, 0)
g.add_edge(2, 3)
g.add_edge(3, 3)
bfs_result = g.bfs(2)
puts "BFS Result: #{bfs_result}"

# Dijkstra's Algorithm
g = Graph.new(5)
g.add_edge(0, 1, 2)
g.add_edge(0, 2, 4)
g.add_edge(1, 2, 1)
g.add_edge(1, 3, 7)
g.add_edge(2, 3, 4)
dijkstra_result = g.dijkstra(0)
puts "Dijkstra's Algorithm Result: #{dijkstra_result}"

# Floyd-Warshall Algorithm
graph = [
  [0, 5, Float::INFINITY, 10],
  [Float::INFINITY, 0, 3, Float::INFINITY],
  [Float::INFINITY, Float::INFINITY, 0, 1],
  [Float::INFINITY, Float::INFINITY, Float::INFINITY, 0]
]
floyd_warshall_result = floyd_warshall(graph)
puts "Floyd-Warshall Algorithm Result:"
floyd_warshall_result.each { |row| puts row }

# Kruskal's Algorithm
g = Graph.new(4)
g.add_edge(0, 1, 10)
g.add_edge(0, 2, 6)
g.add_edge(0, 3, 5)
g.add_edge(1, 3, 15)
g.add_edge(2, 3, 4)
kruskal_result = g.kruskal_mst
puts "Kruskal's Algorithm Result: #{kruskal_result}"

# Topological Sort
graph = {
  0 => [1, 2],
  1 => [3],
  2 => [3],
  3 => []
}
topological_sort_result = topological_sort(graph)
puts "Topological Sort Result: #{topological_sort_result}"

# Knapsack Problem
values = [60, 100, 120]
weights = [10, 20, 30]
capacity = 50
knapsack_result = knapsack_problem(values, weights, capacity)
puts "Knapsack Problem Result: #{knapsack_result}"

# Longest Common Subsequence
X = "AGGTAB"
Y = "GXTXAYB"
lcs_result = longest_common_subsequence(X, Y)
puts "Longest Common Subsequence Result: #{lcs_result}"
