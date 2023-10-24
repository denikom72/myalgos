// Bubble Sort
function bubbleSort(arr) {
    const n = arr.length;
    for (let i = 0; i < n - 1; i++) {
        for (let j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                [arr[j], arr[j + 1]] = [arr[j + 1], arr[j]];
            }
        }
    }
    return arr;
}

// Quick Sort
function quickSort(arr) {
    if (arr.length <= 1) {
        return arr;
    }
    const pivot = arr[0];
    const left = [];
    const right = [];
    for (let i = 1; i < arr.length; i++) {
        if (arr[i] < pivot) {
            left.push(arr[i]);
        } else {
            right.push(arr[i]);
        }
    }
    return [...quickSort(left), pivot, ...quickSort(right)];
}

// Merge Sort
function mergeSort(arr) {
    if (arr.length <= 1) {
        return arr;
    }
    const mid = Math.floor(arr.length / 2);
    const left = arr.slice(0, mid);
    const right = arr.slice(mid);
    return merge(mergeSort(left), mergeSort(right));
}

function merge(left, right) {
    let result = [];
    let leftIndex = 0;
    let rightIndex = 0;
    while (leftIndex < left.length && rightIndex < right.length) {
        if (left[leftIndex] < right[rightIndex]) {
            result.push(left[leftIndex]);
            leftIndex++;
        } else {
            result.push(right[rightIndex]);
            rightIndex++;
        }
    }
    return result.concat(left.slice(leftIndex), right.slice(rightIndex));
}

// Binary Search
function binarySearch(arr, target) {
    let left = 0;
    let right = arr.length - 1;
    while (left <= right) {
        const mid = Math.floor((left + right) / 2);
        if (arr[mid] === target) {
            return mid;
        }
        if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return -1;
}

// Depth-First Search (DFS)
class Graph {
    constructor(vertices) {
        this.vertices = vertices;
        this.adjacencyList = new Array(vertices).fill(0).map(() => []);
    }

    addEdge(start, end) {
        this.adjacencyList[start].push(end);
    }

    dfs(start) {
        const visited = new Array(this.vertices).fill(false);
        const result = [];
        this.dfsUtil(start, visited, result);
        return result;
    }

    dfsUtil(vertex, visited, result) {
        visited[vertex] = true;
        result.push(vertex);
        for (const neighbor of this.adjacencyList[vertex]) {
            if (!visited[neighbor]) {
                this.dfsUtil(neighbor, visited, result);
            }
        }
    }
}

// Breadth-First Search (BFS)
class Graph {
    constructor(vertices) {
        this.vertices = vertices;
        this.adjacencyList = new Array(vertices).fill(0).map(() => []);
    }

    addEdge(start, end) {
        this.adjacencyList[start].push(end);
    }

    bfs(start) {
        const visited = new Array(this.vertices).fill(false);
        const result = [];
        const queue = [start];
        visited[start] = true;
        while (queue.length > 0) {
            const vertex = queue.shift();
            result.push(vertex);
            for (const neighbor of this.adjacencyList[vertex]) {
                if (!visited[neighbor]) {
                    queue.push(neighbor);
                    visited[neighbor] = true;
                }
            }
        }
        return result;
    }
}

// Dijkstra's Algorithm
class Graph {
    constructor(vertices) {
        this.vertices = vertices;
        this.adjacencyList = new Array(vertices).fill(0).map(() => []);
    }

    addEdge(start, end, weight) {
        this.adjacencyList[start].push({ end, weight });
    }

    dijkstra(start) {
        const dist = new Array(this.vertices).fill(Number.MAX_SAFE_INTEGER);
        dist[start] = 0;
        const visited = [];
        for (let i = 0; i < this.vertices - 1; i++) {
            const u = this.minDistance(dist, visited);
            visited[u] = true;
            for (const neighbor of this.adjacencyList[u]) {
                const v = neighbor.end;
                const weight = neighbor.weight;
                if (!visited[v] && dist[u] !== Number.MAX_SAFE_INTEGER && dist[u] + weight < dist[v]) {
                    dist[v] = dist[u] + weight;
                }
            }
        }
        return dist;
    }

    minDistance(dist, visited) {
        let min = Number.MAX_SAFE_INTEGER;
        let minIndex = -1;
        for (let i = 0; i < this.vertices; i++) {
            if (!visited[i] && dist[i] <= min) {
                min = dist[i];
                minIndex = i;
            }
        }
        return minIndex;
    }
}

// Floyd-Warshall Algorithm
function floydWarshall(graph) {
    const n = graph.length;
    const dist = [...graph];
    for (let k = 0; k < n; k++) {
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                if (dist[i][k] + dist[k][j] < dist[i][j]) {
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }
    return dist;
}

// Kruskal's Algorithm
class Graph {
    constructor(vertices) {
        this.vertices = vertices;
        this.edges = [];
    }

    addEdge(start, end, weight) {
        this.edges.push({ start, end, weight });
    }

    kruskalMST() {
        this.edges.sort((a, b) => a.weight - b.weight);
        const result = [];
        const parent = new Array(this.vertices).fill(-1);
        for (const edge of this.edges) {
            const x = this.find(parent, edge.start);
            const y = this.find(parent, edge.end);
            if (x !== y) {
                result.push(edge);
                this.union(parent, x, y);
            }
        }
        return result;
    }

    find(parent, i) {
        if (parent[i] === -1) {
            return i;
        }
        return this.find(parent, parent[i]);
    }

    union(parent, x, y) {
        const xRoot = this.find(parent, x);
        const yRoot = this.find(parent, y);
        parent[xRoot] = yRoot;
    }
}

// Topological Sort
function topologicalSort(graph) {
    const result = [];
    const visited = new Set();
    const stack = [];
    const dfs = (node) => {
        visited.add(node);
        for (const neighbor of graph[node]) {
            if (!visited.has(neighbor)) {
                dfs(neighbor);
            }
        }
        stack.push(node);
    };
    for (const node of Object.keys(graph)) {
        if (!visited.has(node)) {
            dfs(node);
        }
    }
    while (stack.length > 0) {
        result.push(stack.pop());
    }
    return result;
}

// Knapsack Problem (Dynamic Programming)
function knapsackProblem(values, weights, capacity) {
    const n = values.length;
    const dp = new Array(n + 1).fill(null).map(() => new Array(capacity + 1).fill(0));
    for (let i = 1; i <= n; i++) {
        for (let w = 1; w <= capacity; w++) {
            if (weights[i - 1] <= w) {
                dp[i][w] = Math.max(values[i - 1] + dp[i - 1][w - weights[i - 1]], dp[i - 1][w]);
            } else {
                dp[i][w] = dp[i - 1][w];
            }
        }
    }
    const result = [];
    let i = n;
    let w = capacity;
    while (i > 0 && w > 0) {
        if (dp[i][w] !== dp[i - 1][w]) {
            result.push(i - 1);
            w -= weights[i - 1];
        }
        i--;
    }
    return result.reverse();
}

// Longest Common Subsequence (Dynamic Programming)
function longestCommonSubsequence(x, y) {
    const m = x.length;
    const n = y.length;
    const dp = new Array(m + 1).fill(null).map(() => new Array(n + 1).fill(0));
    for (let i = 1; i <= m; i++) {
        for (let j = 1; j <= n; j++) {
            if (x[i - 1] === y[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
    }
    const lcs = [];
    let i = m;
    let j = n;
    while (i > 0 && j > 0) {
        if (x[i - 1] === y[j - 1]) {
            lcs.push(x[i - 1]);
            i--;
            j--;
        } else if (dp[i - 1][j] > dp[i][j - 1]) {
            i--;
        } else {
            j--;
        }
    }
    return lcs.reverse();
}

// Test the algorithms

// Bubble Sort
const arrBubble = [64, 34, 25, 12, 22, 11, 90];
bubbleSort(arrBubble);
console.log("Bubble Sort Result: " + arrBubble);

// Quick Sort
const arrQuick = [64, 34, 25, 12, 22, 11, 90];
const arrQuickSorted = quickSort(arrQuick);
console.log("Quick Sort Result: " + arrQuickSorted);

// Merge Sort
const arrMerge = [64, 34, 25, 12, 22, 11, 90];
const arrMergeSorted = mergeSort(arrMerge);
console.log("Merge Sort Result: " + arrMergeSorted);

// Binary Search
const arrBinary = [11, 12, 22, 25, 34, 64, 90];
const targetBinary = 25;
const indexBinary = binarySearch(arrBinary, targetBinary);
console.log("Binary Search Result: Element " + targetBinary + " found at index " + indexBinary);

// Depth-First Search (DFS)
const graphDFS = new Graph(4);
graphDFS.addEdge(0, 1);
graphDFS.addEdge(0, 2);
graphDFS.addEdge(1, 2);
graphDFS.addEdge(2, 0);
graphDFS.addEdge(2, 3);
graphDFS.addEdge(3, 3);
const dfsResult = graphDFS.dfs(2);
console.log("DFS Result: " + dfsResult);

// Breadth-First Search (BFS)
const graphBFS = new Graph(4);
graphBFS.addEdge(0, 1);
graphBFS.addEdge(0, 2);
graphBFS.addEdge(1, 2);
graphBFS.addEdge(2, 0);
graphBFS.addEdge(2, 3);
graphBFS.addEdge(3, 3);
const bfsResult = graphBFS.bfs(2);
console.log("BFS Result: " + bfsResult);

// Dijkstra's Algorithm
const graphDijkstra = new Graph(5);
graphDijkstra.addEdge(0, 1, 2);
graphDijkstra.addEdge(0, 2, 4);
graphDijkstra.addEdge(1, 2, 1);
graphDijkstra.addEdge(1, 3, 7);
graphDijkstra.addEdge(2, 3, 4);
const dijkstraResult = graphDijkstra.dijkstra(0);
console.log("Dijkstra's Algorithm Result: " + dijkstraResult);

// Floyd-Warshall Algorithm
const graphFloydWarshall = [
    [0, 5, Infinity, 10],
    [Infinity, 0, 3, Infinity],
    [Infinity, Infinity, 0, 1],
    [Infinity, Infinity, Infinity, 0]
];
const floydWarshallResult = floydWarshall(graphFloydWarshall);
console.log("Floyd-Warshall Algorithm Result:");
for (const row of floydWarshallResult) {
    console.log(row.join(", "));
}

// Kruskal's Algorithm
const graphKruskal = new Graph(4);
graphKruskal.addEdge(0, 1, 10);
graphKruskal.addEdge(0, 2, 6);
graphKruskal.addEdge(0, 3, 5);
graphKruskal.addEdge(1, 3, 15);
graphKruskal.addEdge(2, 3, 4);
const kruskalResult = graphKruskal.kruskalMST();
console.log("Kruskal's Algorithm Result: " + JSON.stringify(kruskalResult));

// Topological Sort
const graphTopological = {
    0: [1, 2],
    1: [3],
    2: [3],
    3: []
};
const topologicalSortResult = topologicalSort(graphTopological);
console.log("Topological Sort Result: " + topologicalSortResult);

// Knapsack Problem
const valuesKnapsack = [60, 100, 120];
const weightsKnapsack = [10, 20, 30];
const capacityKnapsack = 50;
const knapsackResult = knapsackProblem(valuesKnapsack, weightsKnapsack, capacityKnapsack);
console.log("Knapsack Problem Result: " + JSON.stringify(knapsackResult));

// Longest Common Subsequence
const X = "AGGTAB";
const Y = "GXTXAYB";
const lcsResult = longestCommonSubsequence(X, Y);
console.log("Longest Common Subsequence Result: " + lcsResult.join(""));
