def removeElement(nums, val):
    """
    :type nums: List[int]
    :type val: int
    :rtype: int
    """
    new_nums = []
    for num in nums:
        if num != val:
            new_nums.append(num)
    nums = new_nums
    k = len(nums)

    return nums


# print(removeElement([0,1,2,2,3,0,4,2], 2))


a = [1, 2, 1, 3, 3, 1]

print(removeElement(a, 3))
