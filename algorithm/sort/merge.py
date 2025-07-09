class MergeSort:
    def __init__(self):
        pass

    def _merge(self, arr, left, mid, right):
        tmp = [0] * (right - left + 1)
        # i, j, k are the indices for the left, right, and temp arrays, like the pointer of the array
        i, j, k = left, mid + 1, 0
        while i <= mid and j <= right:
            if arr[i] <= arr[j]:
                tmp[k] = arr[i]
                i += 1
            else:
                tmp[k] = arr[j]
                j += 1
            k += 1
        while i <= mid:
            tmp[k] = arr[i]
            i += 1
            k += 1
        while j <= right:
            tmp[k] = arr[j]
            j += 1
            k += 1
        for k in range(len(tmp)):
            arr[left + k] = tmp[k]

    def _merge_sort(self, arr, left, right):
        if left >= right:
            return
        mid = (left + right) // 2
        self._merge_sort(arr, left, mid)
        self._merge_sort(arr, mid + 1, right)
        self._merge(arr, left, mid, right)

    def sort(self, arr):
        if not arr:
            return arr
        self._merge_sort(arr, 0, len(arr) - 1)
        return arr
