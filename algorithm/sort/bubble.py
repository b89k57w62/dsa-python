def bubble_sort_with_flag(arr):
    """
    Time Complexity: O(n^2)
    Space Complexity: O(1)
    """
    n = len(arr)
    flag = False
    for i in range(n - 1, 0, -1):
        for j in range(i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                flag = True
        if not flag:
            break


# 穩定排序，保證相同元素的相對位置
# 適用於小規模資料
# 最佳時間複雜度: O(n)
