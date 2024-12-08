# leetcode easy Merge Sorted Array
class Solution(object):
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: None Do not return anything, modify nums1 in-place instead.
        """
        nums1[m:] = nums2[:n]
        nums1.sort()


# leetcode easy Remove Element
class Solution(object):
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        while val in nums:
            if val in nums:
                nums.remove(val)
        k = len(nums)
        return k


# leetcode easy Remove Duplicates from Sorted Array
class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        unique_ptr = 0
        for i in range(1, len(nums)):
            if nums[i] != nums[unique_ptr]:
                unique_ptr += 1
                nums[unique_ptr] = nums[i]

        return unique_ptr + 1


# leetcode easy Majority Element
class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        for num in nums:
            if nums.count(num) > n / 2:
                return num


# leetcode medium Remove Duplicates from Sorted Array II
class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        j = 2
        for i in range(2, len(nums)):
            if nums[i] != nums[j - 2]:
                nums[j] = nums[i]
                j += 1
        return j


# leetcode medium Rotate Array
class Solution(object):
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        k = k % n
        self.reverse(nums, 0, n - 1)
        self.reverse(nums, 0, k - 1)
        self.reverse(nums, k, n - 1)

    def reverse(self, nums, start, end):
        while start < end:
            nums[start], nums[end] = nums[end], nums[start]
            start += 1
            end -= 1


# leetcode easy Best Time to Buy and Sell Stock
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        max_profit = 0
        min_price = float("inf")
        for price in prices:
            if price < min_price:
                min_price = price
            elif price - min_price > max_profit:
                max_profit = price - min_price
        return max_profit


# leetcode medium Best Time to Buy and Sell Stock II
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        min_price = float("inf")
        max_profit = 0
        total = 0
        for i, price in enumerate(prices):
            if price < min_price:
                min_price = price
            if price - min_price > max_profit:
                max_profit = price - min_price
                total += max_profit
                min_price = float("inf")
            elif i < len(prices) - 1 and price < prices[i + 1]:
                total += prices[i + 1] - price
                min_price = float("inf")
        return total


# leetcode medium Jump Game
class Solution(object):
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        max_range = 0
        for i in range(len(nums)):
            if i > max_range:
                return False
            max_range = max(max_range, i + nums[i])
            if max_range >= len(nums) - 1:
                return True
        return False


# leetcode medium Jump Game II
class Solution(object):
    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        boundary = 0
        counts = 0
        max_distance = 0
        for i in range(len(nums) - 1):
            max_distance = max(max_distance, i + nums[i])
            if i == boundary:
                counts += 1
                boundary = max_distance
        return counts


# leetcode medium H-Index
class Solution(object):
    def hIndex(self, citations):
        """
        :type citations: List[int]
        :rtype: int
        """
        sorted_citations = sorted(citations, reverse=True)
        h_index = 0
        for i, citation in enumerate(sorted_citations):
            if citation >= i + 1:
                h_index = i + 1
            else:
                break
        return h_index


# leetcode easy Ransom Note
class Solution(object):
    def canConstruct(self, ransomNote, magazine):
        """
        :type ransomNote: str
        :type magazine: str
        :rtype: bool
        """
        dd = {}
        for char in magazine:
            if char not in dd:
                dd[char] = 1
            else:
                dd[char] += 1

        for char in ransomNote:
            if char in dd and dd[char] > 0:
                dd[char] -= 1
            else:
                return False
        return True


# leetcode easy Isomorphic Strings
class Solution(object):
    def isIsomorphic(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        if len(s) != len(t):
            return False
        map_s_t = {}
        map_t_s = {}
        for c1, c2 in zip(s, t):
            if (c1 in map_s_t and map_s_t[c1] != c2) or (
                c2 in map_t_s and map_t_s[c2] != c1
            ):
                return False
            map_s_t[c1] = c2
            map_t_s[c2] = c1
        return True


# leetcode easy Word Pattern
class Solution(object):
    def wordPattern(self, pattern, s):
        """
        :type pattern: str
        :type s: str
        :rtype: bool
        """
        list_pattern = [char for char in pattern]
        list_s = s.split(" ")
        if len(list_pattern) != len(list_s):
            return False
        map_pattern_s = {}
        map_s_pattern = {}
        for c1, c2 in zip(list_pattern, list_s):
            if (c1 in map_pattern_s and map_pattern_s[c1] != c2) or (
                c2 in map_s_pattern and map_s_pattern[c2] != c1
            ):
                return False
            map_pattern_s[c1] = c2
            map_s_pattern[c2] = c1
        return True
