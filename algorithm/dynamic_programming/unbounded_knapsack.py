def unbounded_knapsack_dp(wgt: list[int], val: list[int], capacity: int):
    """
    unbounded knapsack problem solver using dynamic programming.
    the difference between 0/1 knapsack and unbounded knapsack is that
    in 0/1 knapsack, each item can be used only once, while in unbounded knapsack,
    each item can be used unlimited times.
    time complexity: O(n*c)
    space complexity: O(n*c)
    """
    n = len(wgt)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for c in range(1, capacity + 1):
            if wgt[i - 1] > c:
                dp[i][c] = dp[i - 1][c]
            else:
                dp[i][c] = max(dp[i - 1][c], dp[i][c - wgt[i - 1]] + val[i - 1])
    return dp[n][capacity]


def coin_change_dp(coins: list[int], amount: int):
    """
    coin change problem solver using dynamic programming.
    time complexity: O(n*amount)
    space complexity: O(amount)
    index i means the amount of money to make change for
    dp[i] means the minimum number of coins needed to make change for the amount i
    """
    dp = [float("inf")] * (amount + 1)
    dp[0] = 0
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    return dp[amount] if dp[amount] != float("inf") else -1


def count_coin_combinations(coins: list[int], amount: int):
    """
    time complexity: O(n*amount)
    space complexity: O(amount)
    index i means the amount of money to make change for
    dp[i] means the number of ways to make change for the amount i
    """
    dp = [0] * (amount + 1)
    dp[0] = 1
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]
    return dp[amount]
