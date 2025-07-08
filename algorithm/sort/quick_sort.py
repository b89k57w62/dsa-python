class QuickSort:
    """
    Quick Sort implementation using median-of-three pivot selection.
    Time Complexity: O(nlogn)
    Space Complexity: O(logn)
    """

    def __init__(self):
        pass

    def _median_of_three(self, arr, left, mid, right):
        """
        Find the median of three values for better pivot selection.
        """
        left_val, mid_val, right_val = arr[left], arr[mid], arr[right]
        if left_val <= mid_val <= right_val or left_val >= mid_val >= right_val:
            return mid
        if mid_val <= left_val <= right_val or mid_val >= left_val >= right_val:
            return left
        return right

    def _partition(self, arr, left, right):
        """
        Partition the array around a pivot element.
        """
        mid = self._median_of_three(arr, left, (left + right) // 2, right)
        arr[left], arr[mid] = arr[mid], arr[left]
        i, j = left, right
        while i < j:
            while i < j and arr[i] <= arr[left]:
                i += 1
            while i < j and arr[j] >= arr[left]:
                j -= 1
            arr[i], arr[j] = arr[j], arr[i]
        arr[i], arr[left] = arr[left], arr[i]
        return i

    def _quick_sort_recursive(self, arr, left, right):
        """
        Recursive helper method for quick sort.
        """
        if left >= right:
            return
        pivot = self._partition(arr, left, right)
        self._quick_sort_recursive(arr, left, pivot - 1)
        self._quick_sort_recursive(arr, pivot + 1, right)

    def sort(self, arr):
        """
        Public method to sort an array using quick sort.

        Args:
            arr: List to be sorted (modified in-place)

        Returns:
            None (sorts in-place)
        """
        if not arr or len(arr) <= 1:
            return
        self._quick_sort_recursive(arr, 0, len(arr) - 1)
