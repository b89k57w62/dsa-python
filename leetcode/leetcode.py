from typing import List, Optional
from collections import deque, Counter
import collections
import operator
import heapq
import math
from utils.tree_node import TreeNode
from utils.list_node import ListNode
from utils.interval import Interval


class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


# leetcode easy 88. Merge Sorted Array
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


# leetcode medium 45. Jump Game II
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


# leetcode medium 274. H-Index
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


# leetcode easy 383. Ransom Note
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


# leetcode easy 290. Word Pattern
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


# leetcode easy 509. Fibonacci Number
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


# leetcode easy 1071. Greatest Common Divisor of Strings
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


# leetcode easy 13. Roman to Integer
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


# leetcode easy 58. Length of Last Word
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


# leetcode easy 14. Longest Common Prefix
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
        seen = {}
        for idx, num in enumerate(nums):
            if target - num in seen:
                return [idx, seen[target - num]]
            seen[num] = idx


# leetcode easy 202. Happy Number
class Solution(object):
    def isHappy(self, n):
        seen = {}
        while n != 1 and n not in seen:
            seen[n] = n
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


# leetcode easy 20. Valid Parentheses
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        pair = {")": "(", "]": "[", "}": "{"}
        for i in s:
            if i in pair.values():
                stack.append(i)
            elif stack and stack[-1] == pair[i]:
                stack.pop()
            else:
                return False
        return not stack


# leetcode easy 155. Min Stack
class MinStack:

    def __init__(self):
        self.stack = []

    def push(self, val: int) -> None:
        if not self.stack:
            self.stack.append((val, val))
        else:
            cur_min = self.stack[-1][1]
            self.stack.append((val, min(val, cur_min)))

    def pop(self) -> None:
        num = self.stack[-1]
        self.stack.pop()
        return num

    def top(self) -> int:
        return self.stack[-1][0]

    def getMin(self) -> int:
        return self.stack[-1][1]


# leetcode medium 150. Evaluate Reverse Polish Notation
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        self.stack = []
        operators = {
            "+": operator.add,
            "-": operator.sub,
            "*": operator.mul,
            "/": lambda a, b: int(a / b),
        }
        for token in tokens:
            if token in operators:
                right = self.stack.pop()
                left = self.stack.pop()
                self.stack.append(operators[token](left, right))
            else:
                self.stack.append(int(token))
        return self.stack.pop()


# leetcode medium 232. Implement Queue using Stacks
class MyQueue:

    def __init__(self):
        self.in_stack = []
        self.out_stack = []

    def push(self, x: int) -> None:
        self.in_stack.append(x)

    def pop(self) -> int:
        self._move()
        return self.out_stack.pop()

    def peek(self) -> int:
        self._move()
        return self.out_stack[-1]

    def empty(self) -> bool:
        return not self.out_stack and not self.in_stack

    def _move(self):
        if not self.out_stack:
            while self.in_stack:
                self.out_stack.append(self.in_stack.pop())


# leetcode medium 933. Number of Recent Calls
class RecentCounter:

    def __init__(self):
        self.queue = deque()

    def ping(self, t: int) -> int:
        self.queue.append(t)
        boundry = t - 3000
        while self.queue and self.queue[0] < boundry:
            self.queue.popleft()
        return len(self.queue)


# leetcode medium 641. Design Circular Deque
class MyCircularDeque:

    def __init__(self, k: int):
        self.nums = [0] * k
        self.front = 0
        self.size = 0

    def _index(self, i: int):
        return i % len(self.nums)

    def insertFront(self, value: int) -> bool:
        if self.size == len(self.nums):
            return False
        self.front = self._index(self.front - 1)
        self.nums[self.front] = value
        self.size += 1
        return True

    def insertLast(self, value: int) -> bool:
        if self.size == len(self.nums):
            return False
        last = self._index(self.front + self.size)
        self.nums[last] = value
        self.size += 1
        return True

    def deleteFront(self) -> bool:
        if self.size == 0:
            return False
        num = self.nums[self.front]
        self.front = self._index(self.front + 1)
        self.size -= 1
        return True

    def deleteLast(self) -> bool:
        if self.size == 0:
            return False
        last = self.front + self.size - 1
        self.size -= 1
        return True

    def getFront(self) -> int:
        if self.isEmpty():
            return -1
        else:
            return self.nums[self.front]

    def getRear(self) -> int:
        if self.isEmpty():
            return -1
        last = self._index(self.front + self.size - 1)
        return self.nums[last]

    def isEmpty(self) -> bool:
        return self.size == 0

    def isFull(self) -> bool:
        return self.size == len(self.nums)


# leetcode medium 622. Design Circular Queue
class MyCircularQueue:

    def __init__(self, k: int):
        self.queue = [0] * k
        self._size = 0
        self._front = 0

    def _index(self, i):
        return i % len(self.queue)

    def enQueue(self, value: int) -> bool:
        if self.isFull():
            return False
        rear = self._index(self._front + self._size)
        self.queue[rear] = value
        self._size += 1
        return True

    def deQueue(self) -> bool:
        if self.isEmpty():
            return False
        self._front = self._index(self._front + 1)
        self._size -= 1
        return True

    def Front(self) -> int:
        return -1 if self.isEmpty() else self.queue[self._front]

    def Rear(self) -> int:
        return (
            -1
            if self.isEmpty()
            else self.queue[self._index(self._front + self._size - 1)]
        )

    def isEmpty(self) -> bool:
        return self._size == 0

    def isFull(self) -> bool:
        return self._size >= len(self.queue)


# leetcode easy 217. Contains Duplicate
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        hash = {}
        for num in nums:
            if num not in hash:
                hash[num] = num
            else:
                return True
        return False


# leetcode easy 242. Valid Anagram
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        hash = {}
        for char in s:
            if char not in hash:
                hash[char] = 1
            else:
                hash[char] += 1
        for char in t:
            if char not in hash:
                return False
            else:
                hash[char] -= 1
                if hash[char] < 0:
                    return False
        return True


# leetcode medium 49. Group Anagrams
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        hash = {}
        for s in strs:
            counters = [0] * 26
            for char in s:
                counters[ord(char) - ord("a")] += 1
            key = tuple(counters)

            if key not in hash:
                hash[key] = []
            hash[key].append(s)
        return list(hash.values())


# leetcode medium 347. Top K Frequent Elements
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        hash = {}
        for num in nums:
            if num not in hash:
                hash[num] = 1
            else:
                hash[num] += 1
        items = list(hash.items())
        items.sort(key=lambda x: x[1], reverse=True)
        return [num for num, count in items[:k]]


# leetcode medium 560. Subarray Sum Equals K
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        past_prefix_sum_count = {0: 1}
        current_prefix_sum = 0
        res = 0
        for num in nums:
            current_prefix_sum += num
            past_prefix_sum = current_prefix_sum - k
            res += past_prefix_sum_count.get(past_prefix_sum, 0)
            past_prefix_sum_count[current_prefix_sum] = (
                past_prefix_sum_count.get(current_prefix_sum, 0) + 1
            )
        return res


# leetcode easy 705. Design HashSet
class MyHashSet:

    def __init__(self):
        self._capacity = 769
        self.buckets = [[] for _ in range(self._capacity)]

    def hash_func(self, key):
        return key % self._capacity

    def add(self, key: int) -> None:
        target_bucket = self.buckets[self.hash_func(key)]
        for item in target_bucket:
            if item == key:
                return
        target_bucket.append(key)

    def remove(self, key: int) -> None:
        target_bucket = self.buckets[self.hash_func(key)]
        for i, item in enumerate(target_bucket):
            if item == key:
                target_bucket.pop(i)

    def contains(self, key: int) -> bool:
        target_bucket = self.buckets[self.hash_func(key)]
        if key in target_bucket:
            return True
        return False


# leetcode easy 706. Design HashMap
class MyHashMap:

    def __init__(self):
        self._capacity = 769
        self._buckets = [[] for _ in range(self._capacity)]

    def hash_func(self, key):
        return key % self._capacity

    def put(self, key: int, value: int) -> None:
        target_bucket = self._buckets[self.hash_func(key)]
        for item in target_bucket:
            if item[0] == key:
                item[1] = value
                return
        target_bucket.append([key, value])

    def get(self, key: int) -> int:
        target_bucket = self._buckets[self.hash_func(key)]
        for item in target_bucket:
            if item[0] == key:
                return item[1]
        return -1

    def remove(self, key: int) -> None:
        target_bucket = self._buckets[self.hash_func(key)]
        for i, item in enumerate(target_bucket):
            if item[0] == key:
                target_bucket.pop(i)


# leetcode easy 205. Isomorphic Strings
class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        hash_1, hash_2 = {}, {}
        for char_1, char_2 in zip(s, t):
            if char_1 not in hash_1 and char_2 not in hash_2:
                hash_1[char_1] = char_2
                hash_2[char_2] = char_1
            else:
                if hash_1.get(char_1) != char_2 or hash_2.get(char_2) != char_1:
                    return False
        return True


# leetcode easy 100. Same Tree


class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p and not q:
            return True
        if not p or not q or p.val != q.val:
            return False
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)


# leetcode easy 101. Symmetric Tree
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        if root is None:
            return True
        return self.isMirror(root.left, root.right)

    def isMirror(self, left, right):
        if not left and not right:
            return True
        elif not left or not right or left.val != right.val:
            return False
        return self.isMirror(left.left, right.right) and self.isMirror(
            left.right, right.left
        )


# leetcode easy 104. Maximum Depth of Binary Tree
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))


# leetcode easy 110. Balanced Binary Tree
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        return self.dfs(root) >= 0

    def dfs(self, root):
        if root is None:
            return 0
        left_height = self.dfs(root.left)
        if left_height < 0:
            return -1
        right_height = self.dfs(root.right)
        if right_height < 0:
            return -1
        if abs(left_height - right_height) > 1:
            return -1
        return max(left_height, right_height) + 1


# leetcode medium 98. Validate Binary Search Tree
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        return self.helper(root, float("-inf"), float("inf"))

    def helper(self, node, low, high):
        if node is None:
            return True
        if node.val <= low or node.val >= high:
            return False
        return self.helper(node.left, low, node.val) and self.helper(
            node.right, node.val, high
        )


# leetcode easy 700. Search in a Binary Search Tree
class Solution:
    def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        current_node = root
        while current_node is not None:
            if current_node.val < val:
                current_node = current_node.right
            elif current_node.val > val:
                current_node = current_node.left
            else:
                return current_node
        return None


# leetcode easy 108. Convert Sorted Array to Binary Search Tree
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        return self._build(0, len(nums) - 1, nums)

    def _build(self, low, high, nums):
        if low > high:
            return None
        mid = (low + high) // 2
        node = TreeNode(nums[mid])
        node.left = self._build(low, mid - 1, nums)
        node.right = self._build(mid + 1, high, nums)
        return node


# leetcode medium 102. Binary Tree Level Order Traversal
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        queue = deque()
        queue.append(root)
        res = []
        while queue:
            level_vals = []
            for _ in range(len(queue)):
                node = queue.popleft()
                level_vals.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(level_vals)
        return res


# leetcode easy 94. Binary Tree Inorder Traversal
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if root is None:
            return []
        left_val = self.inorderTraversal(root.left)
        mid = [root.val]
        right_val = self.inorderTraversal(root.right)
        return left_val + mid + right_val


# leetcode easy 144. Binary Tree Preorder Traversal
class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if root is None:
            return []
        mid_node = [root.val]
        left_node = self.preorderTraversal(root.left)
        right_node = self.preorderTraversal(root.right)
        return mid_node + left_node + right_node


# leetcode easy 145. Binary Tree Postorder Traversal
class Solution:
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if root is None:
            return []
        left_node = self.postorderTraversal(root.left)
        right_node = self.postorderTraversal(root.right)
        mid_node = [root.val]
        return left_node + right_node + mid_node


# leetcode medium 958. Check Completeness of a Binary Tree
class Solution:
    def isCompleteTree(self, root: Optional[TreeNode]) -> bool:
        if not root:
            return False
        que = deque()
        que.append(root)
        first_none = False
        while que:
            node = que.popleft()
            if node is None:
                first_none = True
            else:
                if first_none:
                    return False
                que.append(node.left)
                que.append(node.right)
        return True


# leetcode medium 919. Complete Binary Tree Inserter
class CBTInserter:
    def __init__(self, root: Optional[TreeNode]):
        self.nodes = []
        self.root = root
        que = deque()
        que.append(root)
        while que:
            node = que.popleft()
            self.nodes.append(node)
            if node.left:
                que.append(node.left)
            if node.right:
                que.append(node.right)

    def insert(self, val: int) -> int:
        node = TreeNode(val)
        i = len(self.nodes)
        self.nodes.append(node)
        parent_idx = (i - 1) // 2
        if not self.nodes[parent_idx].left:
            self.nodes[parent_idx].left = node
        else:
            self.nodes[parent_idx].right = node
        return self.nodes[parent_idx].val

    def get_root(self) -> Optional[TreeNode]:
        return self.root


# leetcode medium 230. Kth Smallest Element in a BST
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        dfs_in_order = self.in_order(root)
        return dfs_in_order[k - 1]

    def in_order(self, root):
        if not root:
            return []
        return self.in_order(root.left) + [root.val] + self.in_order(root.right)


# leetcode medium 701. Insert into a Binary Search Tree
class Solution:
    def insertIntoBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        if not root:
            return TreeNode(val)
        if val < root.val:
            root.left = self.insertIntoBST(root.left, val)
        else:
            root.right = self.insertIntoBST(root.right, val)
        return root


# leetcode medium 450. Delete Node in a BST
class Solution:
    def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        if not root:
            return None
        if key < root.val:
            root.left = self.deleteNode(root.left, key)
        elif key > root.val:
            root.right = self.deleteNode(root.right, key)
        else:
            if not root.left:
                return root.right
            if not root.right:
                return root.left
            temp = root.right
            while temp.left:
                temp = temp.left
            root.val = temp.val
            root.right = self.deleteNode(root.right, temp.val)
        return root


# leetcode medium 1382. Balance a Binary Search Tree
class Solution:
    def balanceBST(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        nodes = []
        self.inorder(root, nodes)
        return self.build(nodes, 0, len(nodes) - 1)

    def inorder(self, root, res):
        if not root:
            return None
        self.inorder(root.left, res)
        res.append(root)
        self.inorder(root.right, res)
        return res

    def build(self, res, lo, hi):
        if lo > hi:
            return
        mid = (lo + hi) // 2
        mid_node = res[mid]
        mid_node.left = self.build(res, lo, mid - 1)
        mid_node.right = self.build(res, mid + 1, hi)
        return mid_node


# leetcode medium 109. Convert Sorted List to Binary Search Tree
class Solution:
    def __init__(self):
        self.head = None

    def sortedListToBST(self, head: Optional[ListNode]) -> Optional[TreeNode]:
        length = 0
        cur = head
        while cur:
            length += 1
            cur = cur.next
        self.head = head
        return self.build(0, length - 1)

    def build(self, lo, hi):
        if lo >= hi:
            return
        mid = (lo + hi) // 2
        left = self.build(lo, mid - 1)
        root = TreeNode(self.head.val)
        root.left = left
        self.head = self.head.next
        right = self.build(mid + 1, hi)
        root.right = right
        return root


# leetcode medium 912. Sort an Array
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        if len(nums) <= 1:
            return nums
        mid = len(nums) // 2
        left = self.sortArray(nums[:mid])
        right = self.sortArray(nums[mid:])
        return self.merge(left, right)

    def merge(self, left, right):
        merged = []
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1
        if i < len(left):
            merged.extend(left[i:])
        else:
            merged.extend(right[j:])
        return merged


# leetcode easy 1046. Last Stone Weight
class Solution:
    def lastStoneWeight(self, stones: List[int]) -> int:
        max_heap = [-num for num in stones]
        heapq.heapify(max_heap)
        while len(max_heap) > 1:
            first_num = -heapq.heappop(max_heap)
            second_num = -heapq.heappop(max_heap)
            if first_num != second_num:
                heapq.heappush(max_heap, -(first_num - second_num))
        return -max_heap[0] if max_heap else 0


# leetcode medium 215. Kth Largest Element in an Array
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        heap = nums[:k]
        heapq.heapify(heap)
        for num in nums[k:]:
            if num > heap[0]:
                heapq.heapreplace(heap, num)
        return heap[0]


# leetcode medium 703. Kth Largest Element in a Stream
class KthLargest:
    def __init__(self, k: int, nums: List[int]):
        self.heap = nums
        self.k = k
        heapq.heapify(self.heap)
        while len(nums) > k:
            heapq.heappop(self.heap)

    def add(self, val: int) -> int:
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, val)
        elif val > self.heap[0]:
            heapq.heapreplace(self.heap, val)
        return self.heap[0]


# leetcode medium 973. K Closest Points to Origin
class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        heap = []
        for x, y in points:
            distance = x * x + y * y
            heapq.heappush(heap, (-distance, [x, y]))
            if len(heap) > k:
                heapq.heappop(heap)
        return [points for distance, points in heap]


# leetcode medium 133. Clone Graph
class Solution:
    def cloneGraph(self, node: Optional["Node"]) -> Optional["Node"]:
        if not node:
            return None
        copy = {}
        return self.dfs(copy, node)

    def dfs(self, hash, cur: "Node"):
        if cur in hash:
            return hash[cur]
        clone_node = Node(cur.val)
        hash[cur] = clone_node
        for neighbor in cur.neighbors:
            clone_node.neighbors.append(self.dfs(hash, neighbor))
        return clone_node


# leetcode medium 207. Course Schedule
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        adjacency_list = [[] for _ in range(numCourses)]
        visited_state = [0] * numCourses
        for course, prerequisite in prerequisites:
            adjacency_list[prerequisite].append(course)

        for course in range(numCourses):
            if visited_state[course] == 0:
                if self.has_cycle(visited_state, adjacency_list, course):
                    return False
        return True

    def has_cycle(self, visited_state, adjacency_list, current_course):
        if visited_state[current_course] == 1:
            return True
        elif visited_state[current_course] == 2:
            return False
        visited_state[current_course] = 1
        for next_course in adjacency_list[current_course]:
            if self.has_cycle(visited_state, adjacency_list, next_course):
                return True
        visited_state[current_course] = 2


# leetcode medium 210. Course Schedule II
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        adjacency_list = [[] for _ in range(numCourses)]
        visted_state = [0] * numCourses
        order = []
        for course, prerequisite in prerequisites:
            adjacency_list[prerequisite].append(course)

        for course in range(numCourses):
            if visted_state[course] == 0:
                if self.has_cycle(visted_state, adjacency_list, course, order):
                    return []
        return order[::-1]

    def has_cycle(self, visted_state, adjacency_list, current_course, order):
        if visted_state[current_course] == 1:
            return True
        elif visted_state[current_course] == 2:
            return False
        visted_state[current_course] = 1
        for next_course in adjacency_list[current_course]:
            if self.has_cycle(visted_state, adjacency_list, next_course, order):
                return True
        visted_state[current_course] = 2
        order.append(current_course)
        return False


# leetcode medium 200. Number of Islands
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid:
            return 0
        rows, cols = len(grid), len(grid[0])
        islands_counts = 0
        for row in range(rows):
            for col in range(cols):
                if grid[row][col] == "1":
                    islands_counts += 1
                    self.dfs(grid, row, col)
        return islands_counts

    def dfs(self, grid, row, col):
        rows, cols = len(grid), len(grid[0])
        if 0 <= row < rows and 0 <= col < cols and grid[row][col] == "1":
            grid[row][col] = "0"
        else:
            return

        self.dfs(grid, row + 1, col)
        self.dfs(grid, row - 1, col)
        self.dfs(grid, row, col + 1)
        self.dfs(grid, row, col - 1)


# leetcode medium 310. Minimum Height Trees
class Solution:
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        if n == 1:
            return [0]
        adjacency_list = [set() for _ in range(n)]
        degrees = [0] * n
        for node_1, node_2 in edges:
            adjacency_list[node_1].add(node_2)
            adjacency_list[node_2].add(node_1)
            degrees[node_1] += 1
            degrees[node_2] += 1
        queue = deque()
        for i in range(n):
            if degrees[i] == 1:
                queue.append(i)

        remaining_nodes = n
        while remaining_nodes > 2:
            level_size = len(queue)
            remaining_nodes -= level_size

            for i in range(level_size):
                leaf = queue.popleft()

                for neighbor in adjacency_list[leaf]:
                    degrees[neighbor] -= 1
                    if degrees[neighbor] == 1:
                        queue.append(neighbor)
        return list(queue)


# leetcode medium 785. Is Graph Bipartite?
class Solution:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        colors = [0] * len(graph)
        for i in range(len(graph)):
            if colors[i] == 0:
                que = deque()
                que.append(i)
                colors[i] = 1

                while que:
                    node = que.popleft()
                    for neighbor in graph[node]:
                        if colors[neighbor] == 0:
                            colors[neighbor] = -colors[node]
                            que.append(neighbor)
                        elif colors[neighbor] == colors[node]:
                            return False
        return True


# leetcode easy 704. Binary Search
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        while left <= right:
            mid_idx = (left + right) // 2
            if nums[mid_idx] > target:
                right = mid_idx - 1
            elif nums[mid_idx] < target:
                left = mid_idx + 1
            else:
                return mid_idx
        return -1


# leetcode medium 34. Find First and Last Position of Element in Sorted Array
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        res = [-1, -1]
        left, right = 0, len(nums) - 1
        left_edge = -1

        while left <= right:
            mid = (left + right) // 2
            if nums[mid] < target:
                left = mid + 1
            elif nums[mid] > target:
                right = mid - 1
            else:
                left_edge = mid
                right = mid - 1

        if left_edge == -1:
            return res
        res[0] = left_edge

        left, right = 0, len(nums) - 1
        right_edge = -1

        while left <= right:
            mid = (left + right) // 2
            if nums[mid] < target:
                left = mid + 1
            elif nums[mid] > target:
                right = mid - 1
            else:
                right_edge = mid
                left = mid + 1

        if right_edge == -1:
            return res

        res[1] = right_edge
        return res


# leetcode medium 797. All Paths from Source to Target
class Solution:
    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        res = []
        target_node = len(graph) - 1
        self.dfs(graph, 0, [0], target_node, res)
        return res

    def dfs(self, graph, current_node, current_path, target_node, res):
        if current_node == target_node:
            res.append(current_path)
            return

        for neighbor in graph[current_node]:
            self.dfs(graph, neighbor, current_path + [neighbor], target_node, res)


# leetcode medium 148. Sort List
class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return head

        slow, fast = head, head.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        right_head = slow.next
        slow.next = None
        left_sort = self.sortList(head)
        right_sort = self.sortList(right_head)

        return self.merge_sort(left_sort, right_sort)

    def merge_sort(self, left, right):
        temp = ListNode(0)
        tail = temp

        while left and right:
            if left.val < right.val:
                tail.next = left
                left = left.next
            else:
                tail.next = right
                right = right.next

            tail = tail.next

        if left:
            tail.next = left
        if right:
            tail.next = right
        return temp.next


# leetcode medium 75. Sort Colors
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        pointer_0 = 0
        pointer_2 = len(nums) - 1
        checking = 0

        while checking <= pointer_2:
            if nums[checking] == 0:
                nums[pointer_0], nums[checking] = nums[checking], nums[pointer_0]
                checking += 1
                pointer_0 += 1
            elif nums[checking] == 2:
                nums[pointer_2], nums[checking] = nums[checking], nums[pointer_2]
                pointer_2 -= 1
            else:
                checking += 1


# leetcode easy 1122. Relative Sort Array
class Solution:
    def relativeSortArray(self, arr1: List[int], arr2: List[int]) -> List[int]:
        counts = [0] * 1001
        res = []
        for num in arr1:
            counts[num] += 1

        for num in arr2:
            res.extend([num] * counts[num])
            counts[num] = 0

        for i in range(1001):
            if counts[i] > 0:
                res.extend([i] * counts[i])
        return res


# leetcode medium 451. Sort Characters By Frequency
class Solution:
    def frequencySort(self, s: str) -> str:
        hash = {}
        res = []
        for char in s:
            if char not in hash:
                hash[char] = 1
            else:
                hash[char] += 1
        sorted_item = sorted(hash.items(), key=lambda item: item[1], reverse=True)
        for char, freq in sorted_item:
            res.append(char * freq)
        return "".join(res)


# leetcode medium 179. Largest Number
class LargerNum(str):
    def __lt__(self, other):
        return self + other > other + self


class Solution:
    def largestNumber(self, nums: List[int]) -> str:
        str_nums = [str(num) for num in nums]
        str_nums.sort(key=LargerNum)
        res = "".join(str_nums)
        return "0" if res[0] == "0" else res


# leetcode medium 56. Merge Intervals
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if len(intervals) <= 1:
            return intervals
        intervals.sort(key=lambda x: x[0])

        merged = []
        for interval in intervals:
            if not merged or interval[0] > merged[-1][1]:
                merged.append(interval)
            else:
                merged[-1][1] = max(merged[-1][1], interval[1])
        return merged


# leetcode medium 15. 3Sum
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        res = []
        nums.sort()
        for idx in range(len(nums)):
            if idx > 0 and nums[idx] == nums[idx - 1]:
                continue
            left_idx, right_idx = idx + 1, len(nums) - 1
            while left_idx < right_idx:
                if nums[left_idx] + nums[right_idx] < -nums[idx]:
                    left_idx += 1
                elif nums[left_idx] + nums[right_idx] > -nums[idx]:
                    right_idx -= 1
                else:
                    res.append([nums[idx], nums[left_idx], nums[right_idx]])
                    while left_idx < right_idx and nums[left_idx] == nums[left_idx + 1]:
                        left_idx += 1
                    while (
                        left_idx < right_idx and nums[right_idx] == nums[right_idx - 1]
                    ):
                        right_idx -= 1

                    left_idx += 1
                    right_idx -= 1
        return res


# leetcode medium 452. Minimum Number of Arrows to Burst Balloons
class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        if len(points) == 0:
            return 0
        points.sort(key=lambda x: x[1])
        counter = 1
        init_position = points[0][1]

        for i in range(1, len(points)):
            current_position = points[i][0]
            if current_position > init_position:
                counter += 1
                init_position = points[i][1]

        return counter


# leetcode medium 238. Product of Array Except Self
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        length = len(nums)
        left_arr = [1] * length
        for i in range(1, length):
            left_arr[i] = left_arr[i - 1] * nums[i - 1]

        right_arr = [1] * length
        for i in range(length - 2, -1, -1):
            right_arr[i] = right_arr[i + 1] * nums[i + 1]

        answer = [1] * length
        for i in range(length):
            answer[i] = left_arr[i] * right_arr[i]

        return answer


# leetcode medium 36. Valid Sudoku
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        for row in range(9):
            row_set = set()
            for col in range(9):
                if board[row][col] != ".":
                    if board[row][col] in row_set:
                        return False
                    row_set.add(board[row][col])

        for col in range(9):
            col_set = set()
            for row in range(9):
                if board[row][col] != ".":
                    if board[row][col] in col_set:
                        return False
                    col_set.add(board[row][col])

        boxes = {}
        for row in range(9):
            for col in range(9):
                box_id = (row // 3, col // 3)
                if board[row][col] != ".":
                    if box_id not in boxes:
                        boxes[box_id] = set()
                    if board[row][col] in boxes[box_id]:
                        return False
                    boxes[box_id].add(board[row][col])
        return True


# leetcode medium 128. Longest Consecutive Sequence
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        num_set = set(nums)
        longest_streak = 0

        for num in num_set:
            if num - 1 not in num_set:
                start = num
                current_streak = 1

                while (start + 1) in num_set:
                    start += 1
                    current_streak += 1

                longest_streak = max(longest_streak, current_streak)
        return longest_streak


# leetcode medium 271. Encode and Decode Strings
class Solution:
    def encode(self, strs: List[str]) -> str:
        encode_strings = []
        for s in strs:
            encode_strings.append(f"{len(s)}#{s}")
        return "".join(encode_strings)

    def decode(self, s: str) -> List[str]:
        decode_strings = []
        idx = 0
        while idx < len(s):
            j = idx
            while s[j] != "#":
                j += 1
            length = int(s[idx:j])
            start_of_str = j + 1
            end_of_str = start_of_str + length
            origin_str = s[start_of_str:end_of_str]
            decode_strings.append(origin_str)
            idx = end_of_str
        return decode_strings


# leetcode easy 35. Search Insert Position
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        while left <= right:
            mid_idx = (left + right) // 2
            if target < nums[mid_idx]:
                right = mid_idx - 1
            elif target > nums[mid_idx]:
                left = mid_idx + 1
            else:
                return mid_idx
        return left


# leetcode medium 322. Coin Change
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        if amount == 0:
            return 0
        dp = [float("inf")] * (amount + 1)
        dp[0] = 0
        for i in range(1, amount + 1):
            for coin in coins:
                if i >= coin:
                    dp[i] = min(dp[i], dp[i - coin] + 1)
        return dp[amount] if dp[amount] != float("inf") else -1


# leetcode easy 125. Valid Palindrome
class Solution:
    def isPalindrome(self, s: str) -> bool:
        left_pointer, right_pointer = 0, len(s) - 1
        while left_pointer <= right_pointer:
            if not s[left_pointer].isalnum():
                left_pointer += 1
                continue

            if not s[right_pointer].isalnum():
                right_pointer -= 1
                continue

            if s[left_pointer].lower() != s[right_pointer].lower():
                return False

            left_pointer += 1
            right_pointer -= 1

        return True


# leetcode medium 167. Two Sum II - Input Array Is Sorted
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        left_pointer, right_pointer = 0, len(numbers) - 1
        while left_pointer <= right_pointer:
            current_sum = numbers[left_pointer] + numbers[right_pointer]

            if current_sum == target:
                return [left_pointer + 1, right_pointer + 1]
            elif current_sum < target:
                left_pointer += 1
            else:
                right_pointer -= 1


# leetcode medium 11. Container With Most Water
class Solution:
    def maxArea(self, height: List[int]) -> int:
        left_idx, right_idx = 0, len(height) - 1
        max_area = 0
        while left_idx < right_idx:
            current_area = (right_idx - left_idx) * min(
                height[left_idx], height[right_idx]
            )
            max_area = max(max_area, current_area)
            if height[left_idx] < height[right_idx]:
                left_idx += 1
            else:
                right_idx -= 1
        return max_area


# leetcode medium 739. Daily Temperatures
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        n = len(temperatures)
        ans = [0] * n
        stack = []
        for idx, temp in enumerate(temperatures):
            while stack and temp > temperatures[stack[-1]]:
                pre_idx = stack.pop()
                ans[pre_idx] = idx - pre_idx

            stack.append(idx)
        return ans


# leetcode medium 22. Generate Parentheses
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        res = []
        self._backtrack("", 0, 0, res, n)
        return res

    def _backtrack(self, current_str, open_count, close_count, res, n):
        if len(current_str) == 2 * n:
            res.append(current_str)
            return

        if open_count < n:
            self._backtrack(current_str + "(", open_count + 1, close_count, res, n)
        if close_count < open_count:
            self._backtrack(current_str + ")", open_count, close_count + 1, res, n)


# leetcode medium 853. Car Fleet
class Solution:
    def carFleet(self, target: int, position: List[int], speed: List[int]) -> int:
        car_fleets = 0
        lead_time = 0
        cars = []
        for i, j in zip(position, speed):
            time_arrivial = (target - i) / j
            cars.append((i, time_arrivial))
        cars.sort(reverse=True)

        for _, time in cars:
            if time > lead_time:
                lead_time = time
                car_fleets += 1
        return car_fleets


# leetcode medium 74. Search a 2D Matrix
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        rows = len(matrix)
        cols = len(matrix[0])
        left, right = 0, rows * cols - 1

        while left <= right:
            mid_idx = (left + right) // 2
            row_idx = mid_idx // cols
            col_idx = mid_idx % cols
            if matrix[row_idx][col_idx] > target:
                right = mid_idx - 1
            elif matrix[row_idx][col_idx] < target:
                left = mid_idx + 1
            else:
                return True
        return False


# leetcode medium 875. Koko Eating Bananas
class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        min_rate, max_rate = 1, max(piles)
        res = max_rate
        while min_rate < max_rate:
            avg_rate = (min_rate + max_rate) // 2
            hours_needed = 0
            for pile in piles:
                hours_needed += math.ceil(pile / avg_rate)
            if hours_needed <= h:
                res = avg_rate
                max_rate = avg_rate
            else:
                min_rate = avg_rate + 1
        return res


# leetcode medium 153. Find Minimum in Rotated Sorted Array
class Solution:
    def findMin(self, nums: List[int]) -> int:
        left, right = 0, len(nums) - 1
        while left < right:
            mid = (right + left) // 2
            if nums[mid] > nums[right]:
                left = mid + 1
            else:
                right = mid
        return nums[left]


# leetcode medium 33. Search in Rotated Sorted Array
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left_idx, right_idx = 0, len(nums) - 1
        while left_idx <= right_idx:
            mid_idx = (left_idx + right_idx) // 2
            if nums[mid_idx] == target:
                return mid_idx
            elif nums[mid_idx] < nums[right_idx]:
                if nums[mid_idx] <= target <= nums[right_idx]:
                    left_idx = mid_idx + 1
                else:
                    right_idx = mid_idx - 1
            else:
                if nums[left_idx] <= target < nums[mid_idx]:
                    right_idx = mid_idx - 1
                else:
                    left_idx = mid_idx + 1
        return -1


# leetcode medium 981. Time Based Key-Value Store
class TimeMap:
    def __init__(self):
        self.bucket = {}

    def set(self, key: str, value: str, timestamp: int) -> None:
        if key not in self.bucket:
            self.bucket[key] = []
        self.bucket[key].append([value, timestamp])

    def get(self, key: str, timestamp: int) -> str:
        if key not in self.bucket:
            return ""
        res = ""
        values = self.bucket[key]
        left_idx, right_idx = 0, len(values) - 1
        while left_idx <= right_idx:
            mid_idx = (left_idx + right_idx) // 2
            mid_value, mid_timestamp = values[mid_idx]
            if mid_timestamp <= timestamp:
                res = mid_value
                left_idx = mid_idx + 1
            else:
                right_idx = mid_idx - 1
        return res


# leetcode medium 3. Longest Substring Without Repeating Characters
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        bucket = {}
        left = 0
        max_len = 0
        for right, char in enumerate(s):
            if char in bucket and bucket[char] >= left:
                left = bucket[char] + 1

            bucket[char] = right
            max_len = max(max_len, (right - left) + 1)
        return max_len


# leetcode medium 424. Longest Repeating Character Replacement
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        count = {}
        max_len = 0
        max_freq = 0
        left = 0
        for right, char in enumerate(s):
            count[char] = count.get(char, 0) + 1
            max_freq = max(max_freq, count[char])
            window_len = right - left + 1
            if window_len - max_freq > k:
                count[s[left]] -= 1
                left += 1
            max_len = max(max_len, right - left + 1)
        return max_len


# leetcode medium 567. Permutation in String
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        if len(s1) > len(s2):
            return False

        s1_map = [0] * 26
        window_map = [0] * 26

        for i in range(len(s1)):
            s1_map[ord(s1[i]) - ord("a")] += 1
            window_map[ord(s2[i]) - ord("a")] += 1
        if s1_map == window_map:
            return True

        for i in range(len(s1), len(s2)):
            window_map[ord(s2[i]) - ord("a")] += 1
            window_map[ord(s2[i - len(s1)]) - ord("a")] -= 1
            if s1_map == window_map:
                return True
        return False


# leetcode easy 206. Reverse Linked List
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        cur = head
        while cur:
            next_tmp = cur.next
            cur.next = prev
            prev = cur
            cur = next_tmp
        return prev


# leetcode easy 21. Merge Two Sorted Lists
class Solution:
    def mergeTwoLists(
        self, list1: Optional[ListNode], list2: Optional[ListNode]
    ) -> Optional[ListNode]:
        temp = ListNode(-1)
        curr = temp
        while list1 and list2:
            if list1.val <= list2.val:
                curr.next = list1
                list1 = list1.next
            else:
                curr.next = list2
                list2 = list2.next
            curr = curr.next

        curr.next = list1 or list2
        return temp.next


# leetcode easy 141. Linked List Cycle
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        if not head or not head.next:
            return False
        fast = slow = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                return True
        return False


# leetcode medium 143. Reorder List
class Solution:
    def reorderList(self, head: Optional[ListNode]) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        if not head and not head.next:
            return
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        second_head = slow.next
        slow.next = None

        prev = None
        curr = second_head
        while curr:
            temp = curr.next
            curr.next = prev
            prev = curr
            curr = temp
        first = head
        second = prev
        while second:
            temp1, temp2 = first.next, second.next
            first.next = second
            second.next = temp1
            first = temp1
            second = temp2


# leetcode medium 19. Remove Nth Node From End of List
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dummy = ListNode(-1)
        dummy.next = head
        slow = fast = dummy
        for i in range(n):
            if fast and fast.next:
                fast = fast.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next
        next_to_removed_node = slow.next.next
        slow.next = next_to_removed_node
        return dummy.next


# leetcode medium 138. Copy List with Random Pointer
class Solution:
    def copyRandomList(self, head: "Optional[Node]") -> "Optional[Node]":
        hash_map = {}
        curr = head
        while curr is not None:
            new_node = Node(curr.val)
            hash_map[curr] = new_node
            curr = curr.next
        curr = head
        while curr is not None:
            new_node = hash_map[curr]
            if curr.next is not None:
                new_node.next = hash_map[curr.next]
            if curr.random is not None:
                new_node.random = hash_map[curr.random]
            curr = curr.next
        return hash_map.get(head)


# leetcode medium 2. Add Two Numbers
class Solution:
    def addTwoNumbers(
        self, l1: Optional[ListNode], l2: Optional[ListNode]
    ) -> Optional[ListNode]:
        carry = 0
        dummy = ListNode(0)
        curr = dummy
        while l1 or l2 or carry:
            l1_val = l1.val if l1 else 0
            l2_val = l2.val if l2 else 0
            total_sum = l1_val + l2_val + carry
            carry = total_sum // 10
            digit = total_sum % 10

            curr.next = ListNode(digit)
            curr = curr.next
            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next
        return dummy.next


# leetcode medium 287. Find the Duplicate Number
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        slow = nums[0]
        fast = nums[nums[0]]
        while slow != fast:
            slow = nums[slow]
            fast = nums[nums[fast]]

        slow = 0

        while slow != fast:
            slow = nums[slow]
            fast = nums[fast]

        return slow


# leetcode medium 146. LRU Cache
class LRUCache:
    class Node:
        def __init__(self, key, val):
            self.key = key
            self.val = val
            self.prev = None
            self.next = None

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.head = Node(0, 0)
        self.tail = Node(0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node):
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _add_to_head(self, node):
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
        node.prev = self.head

    def get(self, key: int) -> int:
        if key in self.cache:
            node = self.cache[key]
            self._remove(node)
            self._add_to_head(node)
            return node.val
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            node = self.cache[key]
            node.val = value
            self._remove(node)
            self._add_to_head(node)
        else:
            new_node = Node(key, value)
            self.cache[key] = new_node
            self._add_to_head(new_node)
            if len(self.cache) > self.capacity:
                lru_node = self.tail.prev
                self._remove(lru_node)
                del self.cache[lru_node.key]


# leetcode easy 226. Invert Binary Tree
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return None
        origin_left = root.left
        origin_right = root.right
        root.left = self.invertTree(origin_right)
        root.right = self.invertTree(origin_left)
        return root


# leetcode easy 543. Diameter of Binary Tree
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        self.diameter = 0
        self._depth(root)

        return self.diameter

    def _depth(self, node):
        if not node:
            return 0
        left_depth = self._depth(node.left)
        right_depth = self._depth(node.right)

        self.diameter = max(self.diameter, left_depth + right_depth)
        return 1 + max(left_depth, right_depth)


# leetcode easy 572. Subtree of Another Tree
class Solution:
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        if not root:
            return False
        if not subRoot:
            return True
        if self._isSameTree(root, subRoot):
            return True
        return self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)

    def _isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]):
        if not p and not q:
            return True
        if not p or not q or p.val != q.val:
            return False
        return self._isSameTree(p.left, q.left) and self._isSameTree(p.right, q.right)


# leetcode medium 235. Lowest Common Ancestor of a Binary Search Tree
class Solution:
    def lowestCommonAncestor(
        self, root: "TreeNode", p: "TreeNode", q: "TreeNode"
    ) -> "TreeNode":
        if p.val > root.val and q.val > root.val:
            return self.lowestCommonAncestor(root.right, p, q)
        elif p.val < root.val and q.val < root.val:
            return self.lowestCommonAncestor(root.left, p, q)
        else:
            return root


# leetcode medium 199. Binary Tree Right Side View
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        res = []
        queue = deque([root])
        while queue:
            level_size = len(queue)
            for i in range(level_size):
                node = queue.popleft()
                if i == level_size - 1:
                    res.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return res


# leetcode medium 1448. Count Good Nodes in Binary Tree
class Solution:
    def goodNodes(self, root: TreeNode) -> int:
        self.count = 0
        self._dfs(root, float("-inf"))
        return self.count

    def _dfs(self, node, curr_max_value):
        if not node:
            return
        if node.val >= curr_max_value:
            self.count += 1
        curr_max_value = max(curr_max_value, node.val)

        self._dfs(node.left, curr_max_value)
        self._dfs(node.right, curr_max_value)


# leetcode medium 105. Construct Binary Tree from Preorder and Inorder Traversal
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        self.preorder_idx = 0
        inorder_map = {val: idx for idx, val in enumerate(inorder)}
        return self._dfs(preorder, inorder_map, 0, len(inorder) - 1)

    def _dfs(self, preorder, inorder_map, in_start, in_end):
        if in_start > in_end:
            return None
        root_val = preorder[self.preorder_idx]
        root = TreeNode(root_val)
        self.preorder_idx += 1

        inorder_root_idx = inorder_map[root_val]
        root.left = self._dfs(preorder, inorder_map, in_start, inorder_root_idx - 1)
        root.right = self._dfs(preorder, inorder_map, inorder_root_idx + 1, in_end)

        return root


# leetcode medium 208. Implement Trie (Prefix Tree)
class TrieNode:
    def __init__(self):
        self.child = {}
        self.is_end_of_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        current_node = self.root
        for char in word:
            if char not in current_node.child:
                current_node.child[char] = TrieNode()
            current_node = current_node.child[char]
        current_node.is_end_of_word = True

    def search(self, word: str) -> bool:
        current_node = self.root
        for char in word:
            if char not in current_node.child:
                return False
            current_node = current_node.child[char]
        return current_node.is_end_of_word

    def startsWith(self, prefix: str) -> bool:
        current_node = self.root
        for char in prefix:
            if char not in current_node.child:
                return False
            current_node = current_node.child[char]
        return True


# leetcode medium 211. Design Add and Search Words Data Structure
class WordDictionary:
    def __init__(self):
        self.root = TrieNode()

    def addWord(self, word: str) -> None:
        curr_node = self.root
        for char in word:
            if char not in curr_node.child:
                curr_node.child[char] = TrieNode()
            curr_node = curr_node.child[char]
        curr_node.is_end_of_word = True

    def _dfs(self, idx, node):
        curr_node = node
        for i in range(idx, len(self.word)):
            char = self.word[i]
            if char == ".":
                for child_node in curr_node.child.values():
                    if self._dfs(i + 1, child_node):
                        return True
                return False
            else:
                if char not in curr_node.child:
                    return False
                curr_node = curr_node.child[char]
        return curr_node.is_end_of_word

    def search(self, word: str) -> bool:
        self.word = word
        return self._dfs(0, self.root)


class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        # place the most frequent tasks first
        # completed on cycles are n + 1
        task_counts = Counter(tasks)  # {task: count}
        max_heap = []
        for count in task_counts.values():
            heapq.heappush(max_heap, -count)
        time = 0
        while max_heap:
            temp_task = []
            idx = 0  # number of tasks completed in the current cycle
            while idx <= n:
                if not max_heap:
                    break
                curr_freq = heapq.heappop(max_heap)
                curr_freq += 1
                idx += 1
                if curr_freq < 0:
                    temp_task.append(curr_freq)
            if temp_task:
                time += n + 1
            else:
                time += idx
            for freq in temp_task:
                heapq.heappush(max_heap, freq)
        return time


# leetcode medium 359. Logger Rate Limiter
class Twitter:

    def __init__(self):
        self.time = 0
        self.tweets = collections.defaultdict(list)
        self.follows = collections.defaultdict(set)

    def postTweet(self, userId: int, tweetId: int) -> None:
        self.time += 1
        self.tweets[userId].append((self.time, tweetId))

    def getNewsFeed(self, userId: int) -> List[int]:
        res = []
        max_heap = []
        self.follows[userId].add(userId)
        for followeeId in self.follows[userId]:
            if self.tweets[followeeId]:
                followee_tweets = self.tweets[followeeId]
                lastest_tweet_idx = len(followee_tweets) - 1
                timestamp, tweetId = followee_tweets[lastest_tweet_idx]
                heapq.heappush(
                    max_heap, (-timestamp, tweetId, followeeId, lastest_tweet_idx)
                )
        while max_heap and len(res) < 10:
            timestamp, tweetId, followeeId, lastest_tweet_idx = heapq.heappop(max_heap)
            res.append(tweetId)
            if lastest_tweet_idx > 0:
                prev_tweet_idx = lastest_tweet_idx - 1
                prev_timestamp, prev_tweetId = self.tweets[followeeId][prev_tweet_idx]
                heapq.heappush(
                    max_heap,
                    (-prev_timestamp, prev_tweetId, followeeId, prev_tweet_idx),
                )
        return res

    def follow(self, followerId: int, followeeId: int) -> None:
        self.follows[followerId].add(followeeId)

    def unfollow(self, followerId: int, followeeId: int) -> None:
        if followeeId in self.follows[followerId]:
            self.follows[followerId].remove(followeeId)


# leetcode medium 57. Insert Interval
class Solution:
    def insert(
        self, intervals: List[List[int]], newInterval: List[int]
    ) -> List[List[int]]:
        res = []
        i = 0
        n = len(intervals)
        while i < n and intervals[i][1] < newInterval[0]:
            res.append(intervals[i])
            i += 1
        while i < n and intervals[i][0] <= newInterval[1]:
            newInterval[0] = min(intervals[i][0], newInterval[0])
            newInterval[1] = max(intervals[i][1], newInterval[1])
            i += 1
        res.append(newInterval)
        while i < n:
            res.append(intervals[i])
            i += 1
        return res


# leetcode medium 435. Non-overlapping Intervals
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        if not intervals:
            return 0
        intervals.sort(key=lambda x: x[1])
        removed_count = 0
        prev_end = intervals[0][1]
        for i in range(1, len(intervals)):
            curr_start = intervals[i][0]
            curr_end = intervals[i][1]
            if prev_end > curr_start:
                removed_count += 1
            else:
                prev_end = curr_end
        return removed_count


# leetcode easy 252. Meeting Rooms
class Solution:
    def canAttendMeetings(self, intervals: List[Interval]) -> bool:
        if not intervals:
            return True
        intervals.sort(key=lambda x: x.start)
        prev_end = intervals[0].end
        for i in range(1, len(intervals)):
            if prev_end > intervals[i].start:
                return False
            prev_end = intervals[i].end
        return True
