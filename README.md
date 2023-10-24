# myalgos
Algos for Golang, Python, Ruby, php8 and Java

    Bubble Sort:
        Use when simplicity is more important than efficiency.
        Not recommended for large datasets due to its O(n^2) time complexity.

    Quick Sort:
        Use for efficient sorting of large datasets.
        Provides an average-case time complexity of O(n log n).

    Merge Sort:
        Use when you need a stable, efficient sorting algorithm.
        Provides consistent O(n log n) time complexity.

    Binary Search:
        Use to search for an element in a sorted array efficiently.
        Has a time complexity of O(log n).

    Depth-First Search (DFS):
        Use to traverse and explore all vertices in a graph.
        Often used for pathfinding and graph analysis.

    Breadth-First Search (BFS):
        Use to find the shortest path in an unweighted graph.
        Also used in web crawling and network routing.

    Dijkstra's Algorithm:
        Use to find the shortest path in a weighted graph with non-negative edges.
        Helpful in navigation and network routing.

    Floyd-Warshall Algorithm:
        Use for finding all shortest paths in a weighted graph.
        Suitable for small graphs or when negative edge weights are allowed.

    Kruskal's Algorithm:
        Use to find the minimum spanning tree in a weighted graph.
        Often used in network design and clustering.

    Prim's Algorithm (Missing from the previous list):
        Use for finding the minimum spanning tree in a weighted graph.
        Similar to Kruskal's algorithm but can be more efficient for dense graphs.

    Breadth-First Search (BFS) (Repeated for clarification):
        Use for finding the shortest path in an unweighted graph.

    A Search Algorithm*:
        Use for pathfinding in a graph or grid with a heuristic function.
        Balances optimality and efficiency in finding the shortest path.

    Topological Sort:
        Use for ordering tasks with dependencies.
        Commonly applied in scheduling and build systems.

    Knapsack Problem (Dynamic Programming):
        Use to find the most valuable combination of items within a weight limit.
        Applied in resource allocation and optimization.

    Longest Common Subsequence (Dynamic Programming):
        Use for finding the longest common subsequence between two sequences.
        Commonly used in text comparison and DNA analysis.

These algorithms are versatile and can be applied in various problem-solving scenarios, depending on the specific requirements of your project or task.


In computer science and algorithm analysis, O(n log n) is a notation used to describe the time complexity of an algorithm. It indicates that the algorithm's runtime grows in a near-linear fashion with the input size, specifically in a logarithmic relationship.

Here's a breakdown of what O(n log n) means:

    n: Represents the size of the input data or the number of elements to be processed by the algorithm.

    log n: Refers to the logarithm of the input size. In this context, it's often the base-2 logarithm, but other bases are sometimes used as well.

    O: Stands for "big O" notation, which is used to provide an upper bound on the algorithm's time complexity. It describes the worst-case scenario.

So, when an algorithm is said to have a time complexity of O(n log n), it means that the runtime of the algorithm grows in a way that's more efficient than O(n^2) (quadratic time) but not as efficient as O(n) (linear time).

Common algorithms with O(n log n) time complexity include:

    Merge Sort
    Heap Sort
    Many efficient searching and sorting algorithms
    Some tree traversal algorithms

In practical terms, O(n log n) is often considered a good balance between efficiency and scalability. It's frequently used for sorting and searching tasks on large datasets.
