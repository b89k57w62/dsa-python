from typing import List, Optional
from collections import deque
import operator
import heapq
from data_structure.basic_array_list import ListNode


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
        hash = {0: 1}
        prefix = 0
        res = 0
        for num in nums:
            prefix += num
            need = prefix - k
            res += hash.get(need, 0)
            hash[prefix] = hash.get(prefix, 0) + 1
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
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


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
        return self.build(0, length)

    def build(self, lo, hi):
        if lo >= hi:
            return
        mid = (lo + hi) // 2
        left = self.build(lo, mid)
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
        return False


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
        right_edge = -1

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
            res.append(list(current_path))
            return

        for neighbor in graph[current_node]:
            self.dfs(graph, neighbor, current_path + [neighbor], target_node, res)
