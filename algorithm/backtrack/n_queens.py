class NQueensSolver:
    """
    time complexity: O(n!)
    space complexity: O(n^2)
    """

    def __init__(self, n: int):
        self.n = n
        self._reset_state()

    def solve(self) -> list[list[list[str]]]:
        self._reset_state()
        self._backtrack(0)
        return self.solutions

    def _reset_state(self):
        self.board = [["#" for _ in range(self.n)] for _ in range(self.n)]
        self.cols = [False] * self.n
        self.diags1 = [False] * (2 * self.n - 1)
        self.diags2 = [False] * (2 * self.n - 1)
        self.solutions = []

    def _backtrack(self, row: int):
        if row == self.n:
            self.solutions.append([list(row) for row in self.board])
            return

        for col in range(self.n):
            if self._is_valid_position(row, col):
                self._place_queen(row, col)
                self._backtrack(row + 1)
                self._remove_queen(row, col)

    def _is_valid_position(self, row: int, col: int) -> bool:
        diag1 = row - col + self.n - 1
        diag2 = row + col
        return not (self.cols[col] or self.diags1[diag1] or self.diags2[diag2])

    def _place_queen(self, row: int, col: int):
        diag1 = row - col + self.n - 1
        diag2 = row + col

        self.board[row][col] = "Q"
        self.cols[col] = self.diags1[diag1] = self.diags2[diag2] = True

    def _remove_queen(self, row: int, col: int):
        diag1 = row - col + self.n - 1
        diag2 = row + col

        self.board[row][col] = "#"
        self.cols[col] = self.diags1[diag1] = self.diags2[diag2] = False


def n_queens(n: int) -> list[list[list[str]]]:
    solver = NQueensSolver(n)
    return solver.solve()
