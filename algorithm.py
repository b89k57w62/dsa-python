# 題目1: 找出數字0連續出現最多的次數, easy
def find_sequence_zero(n: int):
    """
    問題: 給定一個整數n, 傳回數字0在n中連續出現最多的次數
    範例: n = 9003, 回傳值2
    """
    count = 0
    max_count = 0
    while n > 0:
        d = n % 10
        if d == 0:
            count += 1
            max_count = max(max_count, count)
        else:
            count = 0
        n = n // 10
    return max_count
