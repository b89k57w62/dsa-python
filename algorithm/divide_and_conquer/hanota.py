class HanotaSolver:
    """
    Tower of Hanoi solver using divide and conquer approach.
    time complexity: O(2^n)
    space complexity: O(n)
    """

    def __init__(self, n: int):
        self.n = n
        self._reset_state()

    def solve(
        self, source: list = None, target: list = None, temp: list = None
    ) -> list:
        """
        Solve the Tower of Hanoi problem.

        Args:
            source: Source rod (if None, will create default)
            target: Target rod (if None, will create default)
            temp: Temporary rod (if None, will create default)

        Returns:
            Target rod with all disks moved
        """
        if source is None:
            source = list(range(self.n, 0, -1))  # [n, n-1, ..., 2, 1]
        if target is None:
            target = []
        if temp is None:
            temp = []

        self.source = source
        self.target = target
        self.temp = temp

        self._dfs(self.n, self.source, self.target, self.temp)
        return self.target

    def _reset_state(self):
        """Reset solver state."""
        self.source = []
        self.target = []
        self.temp = []

    def _dfs(self, n: int, source: list, target: list, temp: list):
        """
        Recursive divide and conquer solution.

        Args:
            n: Number of disks to move
            source: Source rod
            target: Target rod
            temp: Temporary rod
        """
        if n == 1:
            self._move(source, target)
            return

        # Move n-1 disks from source to temp using target
        self._dfs(n - 1, source, temp, target)

        # Move the largest disk from source to target
        self._move(source, target)

        # Move n-1 disks from temp to target using source
        self._dfs(n - 1, temp, target, source)

    def _move(self, source: list, target: list):
        """
        Move one disk from source to target.

        Args:
            source: Source rod
            target: Target rod
        """
        if not source:
            raise ValueError("Cannot move from empty source")

        disk = source.pop()
        target.append(disk)


def hanota(n: int, source: list = None, target: list = None, temp: list = None) -> list:
    """
    Convenience function for solving Tower of Hanoi.

    Args:
        n: Number of disks
        source: Source rod (if None, will create default)
        target: Target rod (if None, will create default)
        temp: Temporary rod (if None, will create default)

    Returns:
        Target rod with all disks moved
    """
    solver = HanotaSolver(n)
    return solver.solve(source, target, temp)
