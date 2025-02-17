from typing import List


# leetcode easy 88. Merge Sorted Array
# 內建排序時間複雜度為O((m+n)log(m+n))線性對數階, 空間複雜度是O(m+n), 最差的情況.
# 雙指針解法時間複雜度為O(m+n)線性階, 空間複雜度是O(1)常數階, 未分配其他空間儲存.
class Solution(object):
    def merge(self, nums1: list, m: int, nums2: list, n: int, solutions: str):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: None Do not return anything, modify nums1 in-place instead.
        """
        if solutions == "buildin_sort":
            nums1[m:] = nums2[:n]
            return nums1.sort()
        elif solutions == "two_pointer":
            pointer1, pointer2 = m - 1, n - 1
            index = m + n - 1
            while pointer1 >= 0 and pointer2 >= 0:
                if nums1[pointer1] > nums2[pointer2]:
                    nums1[index] = nums1[pointer1]
                    pointer1 -= 1
                else:
                    nums1[index] = nums2[pointer2]
                    pointer2 -= 1
                index -= 1
            while pointer2 > 0:
                nums1[index] = nums2[pointer2]
                pointer2 -= 1
                index -= 1
            return nums1


# leetcode easy 27. Remove Element
# 內建的remove方法會遍歷list, 故會造成時間複雜度趨近於O(n^2), while+remove等價於雙層迴圈.
# 空間複雜度則是O(1).
# 雙指針解法則時間複雜度O(n), 空間複雜度O(1).
class Solution(object):
    def removeElement(self, nums: List[int], val: int, solutions: str):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        if solutions == "buildin":
            while val in nums:
                if val in nums:
                    nums.remove(val)
            k = len(nums)
        elif solutions == "two pointer":
            k = 0
            for i in range(len(nums)):
                if nums[i] != val:
                    nums[k] = nums[i]
                    k += 1
        return k


# leetcode easy 26. Remove Duplicates from Sorted Array
# 時間複雜度O(n), 空間複雜度O(1)
# 雙指針解法
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


# leetcode medium 80. Remove Duplicates from Sorted Array II
# 時間複雜度O(n), 空間複雜度O(1)
# 雙指針解法
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


# leetcode easy 169. Majority Element
# 多數投票算法, 時間複雜度O(n), 空間複雜度O(1)
# 排序法, 時間複雜度O(n log n), 空間複雜度O(1)
class Solution(object):
    def majorityElement(self, nums, solutions: str):
        """
        :type nums: List[int]
        :rtype: int
        """
        if solutions == "boyer-moore":
            candidate = None
            count = 0
            for num in nums:
                if count == 0:
                    candidate = num
                    count += 1
                elif candidate == num:
                    count += 1
                else:
                    count -= 1
            return candidate
        elif solutions == "sort":
            nums.sort()
            return nums[len(nums) // 2]


# leetcode medium 189. Rotate Array
# 三次反轉法, 時間複雜度O(n), 空間複雜度O(1)
# reverse(), reversed()都是原地修改nums, reversed回傳反向迭代器
class Solution(object):
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        k = k % n  # 針對edge case, 避免k值超過陣列長度
        nums.reverse()
        nums[:k] = reversed(nums[:k])
        nums[k:n] = reversed(nums[k:n])
        return nums


# leetcode easy 121. Best Time to Buy and Sell Stock
# 時間複雜度O(n), 空間複雜度(1)
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


# leetcode medium 122. Best Time to Buy and Sell Stock II
# 時間複雜度O(n), 空間複雜度O(1)
# 貪婪演算法, 局部最優, 無回溯性
class Solution(object):
    def maxProfit(self, prices, solutions: str):
        """
        :type prices: List[int]
        :rtype: int
        """
        total = 0
        if solutions == "標準貪婪演算法":
            for i in range(1, len(prices)):
                if prices[i] > prices[i - 1]:
                    total += prices[i] - prices[i - 1]
        elif solutions == "貪婪演算法但不夠精簡":
            min_price = float("inf")
            for i, price in enumerate(prices):
                if price < min_price:
                    min_price = price
                if i < len(prices) - 1 and price < prices[i + 1]:
                    total += prices[i + 1] - price
                    min_price = float("inf")
        return total


# leetcode medium 55. Jump Game
# 貪婪演算法
# 時間複雜度O(n), 空間複雜度O(1)
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
# 貪婪演算法
# 時間複雜度O(n), 空間複雜度O(1)
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


# leetcode easy Valid Anagram
class Solution(object):
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        if len(s) != len(t):
            return False
        mapping = {}
        mapping_t = {}
        for char in s:
            if char not in mapping:
                mapping[char] = 1
            else:
                mapping[char] += 1

        for char in t:
            if char not in mapping_t:
                mapping_t[char] = 1
            else:
                mapping_t[char] += 1

        for k, v in mapping.items():
            if k not in mapping_t:
                return False
            else:
                if mapping[k] != mapping_t[k]:
                    return False
        return True


# leetcode easy Fibonacci Number
class Solution(object):
    def fib(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n <= 0:
            return 0
        elif n == 1:
            return 1
        result = self.fib(n - 1) + self.fib(n - 2)
        return result


# leetcode easy Greatest Common Divisor of Strings
class Solution(object):
    def gcdOfStrings(self, str1, str2):
        """
        :type str1: str
        :type str2: str
        :rtype: str
        """
        if str1 + str2 != str2 + str1:
            return ""
        gcd_len = self.gcd(len(str2), len(str1) % len(str2))
        return (str1 + str2)[:gcd_len]

    def gcd(self, m, n):
        if n == 0:
            return m
        return self.gcd(n, m % n)


# leetcode easy Roman to Integer
class Solution(object):
    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        benchmark = {
            "I": 1,
            "V": 5,
            "X": 10,
            "L": 50,
            "C": 100,
            "D": 500,
            "M": 1000,
        }
        total = 0
        previous_value = 0
        for char in reversed(s):
            current_value = benchmark[char]
            if current_value >= previous_value:
                total += current_value
            else:
                total -= current_value
            previous_value = current_value
        return total


# leetcode easy Length of Last Word
class Solution(object):
    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        text = s.split(" ")
        result = []
        for world in text:
            if "" == world:
                continue
            else:
                result.append(world)
        return len(result[-1])


# leetcode easy Longest Common Prefix
class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if not strs:
            return ""
        res = strs[0]
        for word in strs[1:]:
            while not word.startswith(res):
                res = res[:-1]
                if not res:
                    return ""
        return res


# leetcode easy 28. Find the Index of the First Occurrence in a String
class Solution(object):
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        if needle in haystack:
            return haystack.index(needle)
        else:
            return -1


# leetcode easy 125. Valid Palindrome
class Solution(object):
    def isPalindrome(self, s, solution):
        """
        :type s: str
        :rtype: bool
        """
        if solution == "tow pointer":
            left, right = 0, len(s) - 1

            while left < right:
                while left < right and not s[left].isalnum():
                    left += 1
                while left < right and not s[right].isalnum():
                    right -= 1
                if s[left].lower() != s[right].lower():
                    return False
                left += 1
                right -= 1
            return True
        else:
            results = []
            for char in s:
                if char.isalpha() or char.isdigit():
                    results.append(char)

            results = ("".join(results)).lower()
            reversed_results = "".join(reversed(results))
            if results == reversed_results:
                return True
            return False


# leetcode easy 392. Is Subsequence
class Solution(object):
    def isSubsequence(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        if len(s) == 0:
            return True
        s_index = 0
        t_index = 0
        while s_index < len(s) and t_index < len(t):
            if s[s_index] == t[t_index]:
                s_index += 1
            t_index += 1
        return s_index == len(s)


# leetcode easy 1. Two Sum
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        storage = {}
        for idx, num in enumerate(nums):
            temp = target - num
            if temp in storage:
                return [idx, storage[temp]]
            storage[num] = idx


# leetcode easy 202. Happy Number
class Solution(object):
    def isHappy(self, n):
        """
        :type n: int
        :rtype: bool
        """
        seen = set()
        while n != 1 and n not in seen:
            seen.add(n)
            n = sum([int(num) ** 2 for num in str(n)])
        return n == 1


# leetcode easy 219. Contains Duplicate II
class Solution(object):
    def containsNearbyDuplicate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        seen = {}
        for idx, num in enumerate(nums):
            if num not in seen:
                seen[num] = idx
            else:
                if abs(seen[num] - idx) <= k:
                    return True
                seen[num] = idx
        return False


# leetcode easy 228. Summary Ranges
class Solution(object):
    def summaryRanges(self, nums):
        """
        :type nums: List[int]
        :rtype: List[str]
        """
        result = []
        if len(nums) == 0:
            return []
        start = nums[0]
        for i in range(1, len(nums)):
            if nums[i] != nums[i - 1] + 1:
                result.append(
                    "{}->{}".format(start, nums[i - 1])
                    if start != nums[i - 1]
                    else "{}".format(start)
                )
                start = nums[i]
        result.append(
            "{}->{}".format(start, nums[-1])
            if start != nums[-1]
            else "{}".format(nums[-1])
        )
        return result
