def count_sort_naive(arr):
    """
    Time Complexity: O(n + k)
    Space Complexity: O(n + k)
    """
    n = max(arr)
    counter = [0] * (n + 1)
    for num in arr:
        counter[num] += 1
    i = 0
    for num in range(n + 1):
        for _ in range(counter[num]):
            arr[i] = num
            i += 1


def count_sort_stable(arr):
    """
    Time Complexity: O(n + k)
    Space Complexity: O(n + k)
    """
    m = max(arr)
    counter = [0] * (m + 1)
    for num in arr:
        counter[num] += 1

    for i in range(m):
        # counter[i] is the number of elements less than or equal to i
        counter[i + 1] += counter[i]
    n = len(arr)
    res = [0] * n
    for i in range(n - 1, -1, -1):
        num = arr[i]
        res[counter[num] - 1] = num
        counter[num] -= 1
    for i in range(n):
        arr[i] = res[i]


# 時間複雜度: O(n + k)
# 空間複雜度: O(n + k)
# 適用場景: 適用於數值範圍較小的排序
# 計數排序只適用於非負整數
