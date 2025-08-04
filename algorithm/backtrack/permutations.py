class PermutationsSolver:
    """
    Given a list of numbers, return all permutations of the list.
    time complexity: O(n!)
    space complexity: O(n^2)
    """

    def __init__(self, nums: list[int]):
        self.nums = nums
        self.state = []
        self.res = []
        self.selected = [False] * len(nums)

    def _backtrack(self):
        if len(self.state) == len(self.nums):
            self.res.append(list(self.state))
            return
        duplicated = set[int]()
        for i, choice in enumerate(self.nums):
            if not self.selected[i] and choice not in duplicated:
                duplicated.add(choice)
                self.selected[i] = True
                self.state.append(choice)
                self._backtrack()
                self.selected[i] = False
                self.state.pop()

    def solve(self):
        self._backtrack()
        return self.res
