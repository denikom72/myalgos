import java.util.*;

// Bubble Sort
class BubbleSort {
    static void bubbleSort(int[] arr) {
        int n = arr.length;
        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < n - i - 1; j++) {
                if (arr[j] > arr[j + 1]) {
                    int temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = temp;
                }
            }
        }
    }
}

// Quick Sort
class QuickSort {
    static void quickSort(int[] arr, int low, int high) {
        if (low < high) {
            int pivot = partition(arr, low, high);
            quickSort(arr, low, pivot - 1);
            quickSort(arr, pivot + 1, high);
        }
    }

    private static int partition(int[] arr, int low, int high) {
        int pivot = arr[high];
        int i = low - 1;
        for (int j = low; j < high; j++) {
            if (arr[j] < pivot) {
                i++;
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
        int temp = arr[i + 1];
        arr[i + 1] = arr[high];
        arr[high] = temp;
        return i + 1;
    }
}

// Merge Sort
class MergeSort {
    static void mergeSort(int[] arr, int left, int right) {
        if (left < right) {
            int mid = (left + right) / 2;
            mergeSort(arr, left, mid);
            mergeSort(arr, mid + 1, right);
            merge(arr, left, mid, right);
        }
    }

    private static void merge(int[] arr, int left, int mid, int right) {
        int n1 = mid - left + 1;
        int n2 = right - mid;

        int[] L = new int[n1];
        int[] R = new int[n2];

        System.arraycopy(arr, left, L, 0, n1);
        System.arraycopy(arr, mid + 1, R, 0, n2);

        int i = 0, j = 0, k = left;

        while (i < n1 && j < n2) {
            if (L[i] <= R[j]) {
                arr[k] = L[i];
                i++;
            } else {
                arr[k] = R[j];
                j++;
            }
            k++;
        }

        while (i < n1) {
            arr[k] = L[i];
            i++;
            k++;
        }

        while (j < n2) {
            arr[k] = R[j];
            j++;
            k++;
        }
    }
}

// Binary Search
class BinarySearch {
    static int binarySearch(int[] arr, int target) {
        int left = 0;
        int right = arr.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (arr[mid] == target) {
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
}

// Depth-First Search (DFS)
class GraphDFS {
    private int vertices;
    private List<List<Integer>> adjacencyList;

    public GraphDFS(int vertices) {
        this.vertices = vertices;
        adjacencyList = new ArrayList<>(vertices);
        for (int i = 0; i < vertices; i++) {
            adjacencyList.add(new ArrayList<>());
        }
    }

    public void addEdge(int start, int end) {
        adjacencyList.get(start).add(end);
    }

    public List<Integer> dfs(int start) {
        List<Integer> result = new ArrayList<>();
        boolean[] visited = new boolean[vertices];
        dfsUtil(start, visited, result);
        return result;
    }

    private void dfsUtil(int vertex, boolean[] visited, List<Integer> result) {
        visited[vertex] = true;
        result.add(vertex);
        for (int neighbor : adjacencyList.get(vertex)) {
            if (!visited[neighbor]) {
                dfsUtil(neighbor, visited, result);
            }
        }
    }
}

import java.util.Arrays;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        // Test Bubble Sort
        int[] bubbleSortArr = {64, 34, 25, 12, 22, 11, 90};
        BubbleSort.bubbleSort(bubbleSortArr);
        System.out.println("Bubble Sort Result: " + Arrays.toString(bubbleSortArr));

        // Test Quick Sort
        int[] quickSortArr = {64, 34, 25, 12, 22, 11, 90};
        QuickSort.quickSort(quickSortArr, 0, quickSortArr.length - 1);
        System.out.println("Quick Sort Result: " + Arrays.toString(quickSortArr));

        // Test Merge Sort
        int[] mergeSortArr = {64, 34, 25, 12, 22, 11, 90};
        MergeSort.mergeSort(mergeSortArr, 0, mergeSortArr.length - 1);
        System.out.println("Merge Sort Result: " + Arrays.toString(mergeSortArr));

        // Test Binary Search
        int[] binarySearchArr = {11, 12, 22, 25, 34, 64, 90};
        int binarySearchResult = BinarySearch.binarySearch(binarySearchArr, 22);
        System.out.println("Binary Search Result (Index): " + binarySearchResult);

        // Test Depth-First Search (DFS)
        GraphDFS graphDFS = new GraphDFS(4);
        graphDFS.addEdge(0, 1);
        graphDFS.addEdge(0, 2);
        graphDFS.addEdge(1, 2);
        graphDFS.addEdge(2, 0);
        graphDFS.addEdge(2, 3);
        graphDFS.addEdge(3, 3);
        List<Integer> dfsResult = graphDFS.dfs(2);
        System.out.println("Depth-First Search Result: " + dfsResult);

        // You can add tests for the remaining algorithms here
    }
}

