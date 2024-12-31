# 題目1: 找出數字0連續出現最多的次數, easy
def find_sequence_zero(n: int) -> int:
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


# 題目2: 判斷是否為回文數字, easy
def is_palindrome_number(n: int, method_type: str) -> bool:
    from math import floor, log

    """
    問題: 給定一個整數n, 判斷此數字是否為回文數字
    範例: 13231, True 
    """
    if method_type == "str":
        num_to_str = str(n)
        reverse_num_to_str = num_to_str[::-1]
        if num_to_str == reverse_num_to_str:
            return True
        return False
    elif method_type == "int":
        digits = floor(log(n, 10)) + 1  # 判斷共幾位數
        loop_count = digits // 2
        for i in range(loop_count):
            low = n % 10
            high = n // (10 ** (digits - 1))
            if low != high:
                return False
            n = n % (10 ** (digits - 1))  # 移除最高位
            n = n // 10  # 移除最低位
            digits -= 2
        return True


# 題目5: star tree -
def build_tree(n: int) -> None:
    """
    問題: 輸入整入, 印出金字塔
    """
    for row in range(1, n + 1):
        sapce = " "
        space_count = n - row
        star_count = 2 * row - 1
        print(f"{space_count * sapce}{"*"*star_count}{space_count * sapce}")


# 題目6: 遞回版歐幾里得演算法
def gcd_euclid(m: int, n: int):
    """
    問題: 求兩數最大公因數, 根據數學定義 gcd(m, n) = gcd(n, m%n)
    """
    if n == 0:
        return m
    return gcd_euclid(n, m % n)


# 題目7: 快速次方
def fastexp(x: int, n: int):
    """
    問題: 回傳值x的n次方, 最多使用2 * log n個乘法
    解法: 快速冪, 核心在每一層都將範圍減半
    """
    if n == 1:
        return x
    half = fastexp(
        x, n // 2
    )  # 暫存結果, 則不需要使用兩次遞迴, 以此避免運算次數以指數增加
    if n % 2 == 0:
        return half * half
    elif n % 2 != 0:
        return half * half * x


# 河內塔
def hanoi(n: int, start: str, end: str, temp: str, steps: list = None):
    if steps == None:
        steps = []
    if n == 1:
        steps.append((start, end))
        return len(steps), steps
    # step1
    hanoi(n - 1, start, temp, end, steps)
    # step2
    steps.append((start, end))
    # step3
    hanoi(n - 1, temp, end, start, steps)
    return len(steps), steps
