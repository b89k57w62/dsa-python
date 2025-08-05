class SubsetSum:
    """
    Subset sum problem solver using backtracking algorithm
    time complexity: O(n^target)
    space complexity: O(target)
    """

    def __init__(self):
        self.state = []
        self.res = []

    def solve_allow_duplicate(self, target: int, choices: list[int]) -> list[list[int]]:
        """
        Find all subsets that sum to target, allowing duplicate choices
        """
        self.res = []
        self.state = []
        choices.sort()  # Sort for optimization
        self._backtrack_allow_duplicate(target, choices, 0)
        return self.res

    def solve_no_duplicate(self, target: int, choices: list[int]) -> list[list[int]]:
        """
        Find all subsets that sum to target, no duplicate choices
        """
        self.res = []
        self.state = []
        choices.sort()  # Sort for optimization and duplicate handling
        self._backtrack_no_duplicate(target, choices, 0)
        return self.res

    def _backtrack_allow_duplicate(self, target: int, choices: list[int], start: int):
        """
        allow duplicate choices
        time complexity: O(n^target)
        space complexity: O(target)
        """
        if target == 0:
            self.res.append(list(self.state))
            return
        for i in range(start, len(choices)):
            if target - choices[i] < 0:
                break
            self.state.append(choices[i])
            self._backtrack_allow_duplicate(target - choices[i], choices, i)
            self.state.pop()

    def _backtrack_no_duplicate(self, target: int, choices: list[int], start: int):
        """
        no duplicate choices
        time complexity: O(n^target)
        space complexity: O(target)
        """
        if target == 0:
            self.res.append(list(self.state))
            return
        for i in range(start, len(choices)):
            if target - choices[i] < 0:
                break
            if i > start and choices[i] == choices[i - 1]:
                continue
            self.state.append(choices[i])
            self._backtrack_no_duplicate(target - choices[i], choices, i + 1)
            self.state.pop()
