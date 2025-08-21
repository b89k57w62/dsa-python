def coin_change_greedy(coins: list[int], amount: int) -> int:
    """
    coin change problem solver using greedy algorithm.
    time complexity: O(n)
    space complexity: O(1)
    """
    idx = len(coins) - 1
    count = 0
    while amount > 0:
        while idx >= 0 and coins[idx] > amount:
            idx -= 1
        if idx < 0:
            return -1
        amount -= coins[idx]
        count += 1
    return count if amount == 0 else -1
