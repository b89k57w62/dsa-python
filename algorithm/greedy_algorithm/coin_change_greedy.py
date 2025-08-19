def coin_change_greedy(coins: list[int], amount: int) -> int:
    """
    coin change problem solver using greedy algorithm.
    time complexity: O(n)
    space complexity: O(1)
    """
    end_idx = len(coins) - 1
    count = 0
    while amount > 0:
        while end_idx >= 0 and coins[end_idx] > amount:
            end_idx -= 1
        if end_idx < 0:
            return -1
        amount -= coins[end_idx]
        count += 1
    return count if amount == 0 else -1
