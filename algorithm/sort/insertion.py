def insertion_sort(arr):
    """
    Time Complexity: O(n^2)
    Space Complexity: O(1)
    """
    for i in range(1, len(arr)):
        base = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > base:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = base
