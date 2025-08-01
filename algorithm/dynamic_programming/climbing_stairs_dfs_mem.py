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
        if n == 1:
            return 1
        if n == 2:
            return 2

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