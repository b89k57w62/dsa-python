def knapsack_dfs(wgt: list[list], val: list[int], current_index: int, capacity: int):
    """
    knapsack problem solver using dfs.
    time complexity: O(2^n)
    space complexity: O(n)
    current_index represent the current item index to consider
    capacity represent the capacity of the knapsack
    wgt represent the weight of the items
    val represent the value of the items
    return the maximum value of the items that can be put into the knapsack
    """
    if current_index < 0 or capacity == 0:
        return 0
    if wgt[current_index] > capacity:
        return knapsack_dfs(wgt, val, current_index - 1, capacity)
    can_take = (
        knapsack_dfs(wgt, val, current_index - 1, capacity - wgt[current_index])
        + val[current_index]
    )
    cannot_take = knapsack_dfs(wgt, val, current_index - 1, capacity)
    return max(can_take, cannot_take)


def knapsack_dfs_mem(
    wgt: list[list],
    val: list[int],
    current_index: int,
    capacity: int,
    mem: list[list[int]],
):
    """
    knapsack problem solver using dfs with memoization.
    time complexity: O(n*c)
    space complexity: O(n*c)
    """
    if current_index < 0 or capacity == 0:
        return 0
    if mem[current_index][capacity] != -1:
        return mem[current_index][capacity]
    if wgt[current_index] > capacity:
        return knapsack_dfs_mem(wgt, val, current_index - 1, capacity, mem)

    can_take = (
        knapsack_dfs_mem(
            wgt, val, current_index - 1, capacity - wgt[current_index], mem
        )
        + val[current_index]
    )
    cannot_take = knapsack_dfs_mem(wgt, val, current_index - 1, capacity, mem)
    mem[current_index][capacity] = max(can_take, cannot_take)
    return mem[current_index][capacity]


def knapsack_dp(wgt: list[int], val: list[int], capacity: int):
    """
    knapsack problem solver using dynamic programming.
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
                dp[i][c] = max(dp[i - 1][c], dp[i - 1][c - wgt[i - 1]] + val[i - 1])
    return dp[n][capacity]
