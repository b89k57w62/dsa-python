def binary_search_insert(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return left


def binary_search_left_bound(arr, target):
    idx = binary_search_insert(arr, target)
    if idx == len(arr) or arr[idx] != target:
        return -1
    return idx


def binary_search_right_bound(arr, target):
    idx = binary_search_insert(arr, target + 1)
    j = idx - 1
    if j == -1 or arr[j] != target:
        return -1
    return j
