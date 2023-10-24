package main

import (
    "fmt"
    "sort"
)

// Bubble Sort
func bubbleSort(arr []int) {
    n := len(arr)
    for i := 0; i < n-1; i++ {
        for j := 0; j < n-i-1; j++ {
            if arr[j] > arr[j+1] {
                arr[j], arr[j+1] = arr[j+1], arr[j]
            }
        }
    }
}

// Quick Sort
func quickSort(arr []int) []int {
    if len(arr) <= 1 {
        return arr
    }
    pivot := arr[0]
    var left, right []int
    for _, v := range arr[1:] {
        if v < pivot {
            left = append(left, v)
        } else {
            right = append(right, v)
        }
    }
    return append(append(quickSort(left), pivot), quickSort(right)...)
}

// Merge Sort
func mergeSort(arr []int) []int {
    if len(arr) <= 1 {
        return arr
    }
    mid := len(arr) / 2
    left := mergeSort(arr[:mid])
    right := mergeSort(arr[mid:])
    return merge(left, right)
}

func merge(left, right []int) []int {
    result := []int{}
    for len(left) > 0 || len(right) > 0 {
        if len(left) > 0 && len(right) > 0 {
            if left[0] <= right[0] {
                result = append(result, left[0])
                left = left[1:]
            } else {
                result = append(result, right[0])
                right = right[1:]
            }
        } else if len(left) > 0 {
            result = append(result, left[0])
            left = left[1:]
        } else if len(right) > 0 {
            result = append(result, right[0])
            right = right[1:]
        }
    }
    return result
}

// Binary Search
func binarySearch(arr []int, target int) int {
    left, right := 0, len(arr)-1
    for left <= right {
        mid := left + (right-left)/2
        if arr[mid] == target {
            return mid
        }
        if arr[mid] < target {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    return -1
}

// Depth-First Search (DFS)
type Graph struct {
    vertices int
    adj      map[int][]int
}

func (g *Graph) DFS(start int) []int {
    visited := make(map[int]bool)
    result := []int{}
    g.dfsUtil(start, visited, &result)
    return result
}

func (g *Graph) dfsUtil(v int, visited map[int]bool, result *[]int) {
    visited[v] = true
    *result = append(*result, v)
    for _, n := range g.adj[v] {
        if !visited[n] {
            g.dfsUtil(n, visited, result)
        }
    }
}

// Breadth-First Search (BFS)
func (g *Graph) BFS(start int) []int {
    visited := make(map[int]bool)
    result := []int{}
    queue := []int{start}
    visited[start] = true

    for len(queue) > 0 {
        vertex := queue[0]
        queue = queue[1:]
        result = append(result, vertex)

        for _, n := range g.adj[vertex] {
            if !visited[n] {
                visited[n] = true
                queue = append(queue, n)
            }
        }
    }
    return result
}

// Dijkstra's Algorithm
func (g *Graph) Dijkstra(start int) map[int]int {
    dist := make(map[int]int)
    for i := 0; i < g.vertices; i++ {
        dist[i] = int(^uint(0) >> 1) // Max integer value
    }
    dist[start] = 0
    visited := make(map[int]bool)
    for i := 0; i < g.vertices-1; i++ {
        u := g.minDistance(dist, visited)
        visited[u] = true
        for v, weight := range g.graph[u] {
            if !visited[v] && dist[u]+weight < dist[v] {
                dist[v] = dist[u] + weight
            }
        }
    }
    return dist
}

func (g *Graph) minDistance(dist map[int]int, visited map[int]bool) int {
    min := int(^uint(0) >> 1)
    var minIndex int
    for v := 0; v < g.vertices; v++ {
        if !visited[v] && dist[v] <= min {
            min = dist[v]
            minIndex = v
        }
    }
    return minIndex
}

// Floyd-Warshall Algorithm
func floydWarshall(graph [][]int) [][]int {
    n := len(graph)
    dist := make([][]int, n)
    for i := range dist {
        dist[i] = make([]int, n)
        for j := range dist[i] {
            dist[i][j] = graph[i][j]
        }
    }
    for k := 0; k < n; k++ {
        for i := 0; i < n; i++ {
            for j := 0; j < n; j++ {
                if dist[i][k]+dist[k][j] < dist[i][j] {
                    dist[i][j] = dist[i][k] + dist[k][j]
                }
            }
        }
    }
    return dist
}

// Kruskal's Algorithm
type Graph struct {
    vertices int
    edges    []Edge
}

type Edge struct {
    src, dest, weight int
}

func (g *Graph) kruskalMST() []Edge {
    result := make([]Edge, 0)
    sort.Slice(g.edges, func(i, j int) bool {
        return g.edges[i].weight < g.edges[j].weight
    })
    parent := make([]int, g.vertices)
    for i := 0; i < g.vertices; i++ {
        parent[i] = i
    }
    for _, edge := range g.edges {
        if g.isCycle(parent, edge.src, edge.dest) {
            continue
        }
        result = append(result, edge)
        g.union(parent, edge.src, edge.dest)
    }
    return result
}

func (g *Graph) find(parent []int, i int) int {
    if parent[i] == i {
        return i
    }
    return g.find(parent, parent[i])
}

func (g *Graph) union(parent []int, x, y int) {
    rootX := g.find(parent, x)
    rootY := g.find(parent, y)
    parent[rootX] = rootY
}

func (g *Graph) isCycle(parent []int, x, y int) bool {
    rootX := g.find(parent, x)
    rootY := g.find(parent, y)
    return rootX == rootY
}

func main() {
    // You can test the algorithms here
}
