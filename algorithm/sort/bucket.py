def bucket_sort(arr):
    k = len(arr) // 2
    buckets = [[] for _ in range(k)]
    for num in arr:
        # settle the num into the bucket
        idx = int(num * k)
        buckets[idx].append(num)

    for bucket in buckets:
        bucket.sort()

    i = 0
    for bucket in buckets:
        for num in bucket:
            arr[i] = num
            i += 1


# 穩定排序，保證相同元素的相對位置
# 適用於大規模資料
# 最佳時間複雜度: O(n)
# 最差時間複雜度: O(n^2)
# 平均時間複雜度: O(n)
# 空間複雜度: O(n)
