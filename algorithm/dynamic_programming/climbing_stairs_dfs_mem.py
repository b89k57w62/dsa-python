class ClimbingStairsSolver:
    """
    Climbing stairs problem solver using dynamic programming with memoization.
    time complexity: O(n)
    space complexity: O(n)
    """

    def __init__(self, n: int):
        if n <= 0:
            raise ValueError("Number of stairs must be positive")
        self.n = n
        self._reset_state()

    def solve(self) -> int:
        """
        Solve the climbing stairs problem.

        Returns:
            Number of distinct ways to climb to the top
        """
        self._reset_state()
        return self._dfs(self.n)

    def _reset_state(self):
        """Reset solver state and initialize memoization array."""
        self.mem = [0] * (self.n + 1)

    def _dfs(self, n: int) -> int:
        """
        Recursive function with memoization to calculate climbing ways.

        Args:
            n: Number of stairs to climb

        Returns:
            Number of distinct ways to climb n stairs
        """
        # Base cases
        if n == 1 or n == 2:
            return n
        # Check if already computed
        if self.mem[n] != 0:
            return self.mem[n]

        # Recursive relation: f(n) = f(n-1) + f(n-2)
        count = self._dfs(n - 1) + self._dfs(n - 2)
        self.mem[n] = count
        return count


def climbing_stairs_mem(n: int) -> int:
    """
    Convenience function for solving climbing stairs problem.

    Args:
        n: Number of stairs to climb

    Returns:
        Number of distinct ways to climb to the top
    """
    solver = ClimbingStairsSolver(n)
    return solver.solve()


def climbing_stairs_dp(n: int):
    """
    climbing stairs problem solver using dynamic programming.
    time complexity: O(n)
    space complexity: O(n)
    """
    if n == 1 or n == 2:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    dp[2] = 2
    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]


def climbing_stairs_dp_rolling_var(n: int):
    """
    climbing stairs problem solver using dynamic programming with rolling variables.
    time complexity: O(n)
    space complexity: O(1)
    """
    if n == 1 or n == 2:
        return n
    a, b = 1, 2
    for _ in range(3, n + 1):
        a, b = b, a + b
    return b


def min_cost_climbing_stairs_dp(cost: list[int]):
    """
    min cost climbing stairs problem solver using dynamic programming.
    time complexity: O(n)
    space complexity: O(n)
    """
    n = len(cost)
    dp = [0] * (n + 1)
    dp[0] = cost[0]
    dp[1] = cost[1]
    for i in range(2, n):
        dp[i] = min(dp[i - 1], dp[i - 2]) + cost[i]
    return dp[n]


def min_cost_climbing_stairs_dp_rolling_var(cost: list[int]):
    """
    min cost climbing stairs problem solver using dynamic programming with rolling variables.
    time complexity: O(n)
    space complexity: O(1)
    """
    n = len(cost) - 1
    if n == 1 or n == 2:
        return cost[n]
    a, b = cost[0], cost[1]
    for i in range(3, n + 1):
        a, b = b, min(a, b) + cost[i]
    return min(a, b)
