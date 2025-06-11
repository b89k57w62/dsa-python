class CustomList:
    def __init__(self, nums: list[int]):
        self.nums = nums

    def insert(nums: list[int], num: int, index: int):
        for i in range(len(nums) - 1, index, -1):
            nums[i] = nums[i - 1]
        nums[index] = num

    def remove(nums: list[int], index: int):
        for i in range(index, len(nums) - 1):
            nums[i] = nums[i + 1]

    def extend(nums: list[int], enlarge: int):
        res = [0] * (len(nums) + enlarge)
        for i in range(len(nums)):
            res[i] = nums[i]
        return res


nums = CustomList([1, 2, 3, 4, 5])
