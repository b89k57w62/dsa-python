# 6kyu Delete occurrences of an element if it occurs more than n times
def delete_nth(order, max_e):
    result = []
    for num in order:
        if result.count(num) < max_e:
            result.append(num)
    return result


# 6kyu Tribonacci Sequence
def tribonacci(signature, n):
    if n <= 3:
        return signature[:n]
    result = signature[:]
    for i in range(3, n):
        result.append(sum(result[-3:]))
    return result


# 6kyu Bit Counting
def count_bits(n):
    bin_num = format(n, "b")
    return bin_num.count("1")


# 6kyu Who likes it?
def likes(names):
    if len(names) == 0:
        return "no one likes this"
    elif len(names) == 1:
        return f"{names[0]} likes this"
    elif len(names) == 2:
        return f"{names[0]} and {names[1]} like this"
    elif len(names) == 3:
        return f"{names[0]}, {names[1]} and {names[2]} like this"
    return f"{names[0]}, {names[1]} and {len(names)-2} others like this"


# 6kyu Find the odd int
def find_it(seq):
    from collections import Counter

    count_num = Counter(seq)
    for key in count_num:
        if count_num[key] % 2 == 1:
            return key


# 5kyu Simple Pig Latin
def pig_it(text):
    ans = []
    for str in text.split(" "):
        if str.isalpha():
            ans.append(str.replace(str[0], "", 1) + str[0] + "ay")
        else:
            ans.append(str)
    return " ".join(ans)


# 6kyu Array.diff
def array_diff(a, b):
    ans = []
    for num in a:
        if num not in b:
            ans.append(num)
    return ans
    # solution2
    # return [num for num in a if num not in b]


# 7kyu Jaden Casing Strings
def to_jaden_case(string):
    return " ".join([words.capitalize() for words in string.split(" ")])


# 5kyu Not very secure
def alphanumeric(password):
    if len(password) == 0:
        return False
    ans = []
    for char in password:
        if char.isalpha() or char.isnumeric():
            ans.append(char)
    return True if len("".join(ans)) == len(password) else False
    # solution2
    # return password.isalnum()


# 6kyu Stop gninnipS My sdroW!
def spin_words(sentence):
    array = sentence.split(" ")
    return " ".join([words[::-1] if len(words) >= 5 else words for words in array])


# 6kyu Which are in?
def in_array(array1, array2):
    ans = []
    for i in range(len(array1)):
        for words in array2:
            if array1[i] in words:
                ans.append(array1[i])
    return sorted(list(set(ans)))


# 6kyu Break camelCase
def solution(s):
    return "".join([" " + char if char.isupper() else char for char in s])


# 5kyu Pete, the baker
def cakes(recipe, available):
    ans = []
    if recipe.keys() <= available.keys():
        for key in recipe:
            ans.append(available[key] // recipe[key])
        return min(ans)
    return False
    # solution2
    # return min(available.get(k, 0)//recipe[k] for k in recipe)


# 5kyu Human Readable Time
def make_readable(seconds):
    hrs = seconds // 3600
    mins = (seconds % 3600) // 60
    secs = (seconds % 3600) % 60
    return f"{hrs:02}:{mins:02}:{secs:02}"


# 6kyu Equal Sides Of An Array
def find_even_index(arr):
    for i in range(len(arr)):
        if sum(arr[0 : i + 1]) == sum(arr[i:]):
            return i
    return -1


# 6kyu The Supermarket Queue
def queue_time(customers, n):
    if customers == 0:
        return 0
    cashers = [0] * n
    for customer in customers:
        min_index = cashers.index(min(cashers))
        cashers[min_index] += customer
    return max(cashers)


# 6kyu Convert string to camel case
def to_camel_case(text):
    array = [char for char in text.replace("_", "-").split("-")]
    ans = array[0] + "".join([char.capitalize() for char in array if char != array[0]])
    return ans


# 4kyu Sum Strings as Numbers
def sum_strings(x, y):
    x = x.rjust(max(len(x), len(y)), "0")
    y = y.rjust(max(len(x), len(y)), "0")
    res = []
    c = 0
    for num1, num2 in zip(x[::-1], y[::-1]):
        temp = int(num1) + int(num2) + c
        c = temp // 10
        res.append(temp % 10)
    if c:
        res.append(c)
    return "".join(map(str, res[::-1])).lstrip("0") or "0"


# 4kyu Sort binary tree by levels // BFS
def tree_by_levels(node):
    res = []
    queue = []
    if node is not None:
        queue.append(node)
    else:
        return []
    while len(queue) > 0:
        temp = queue.pop(0)
        res.append(temp.value)
        if temp.left is not None:
            queue.append(temp.left)
        if temp.right is not None:
            queue.append(temp.right)
    return res


# 4kyu Connect Four
def who_is_winner(pieces_position_list):
    matrix = [[0] * 7 for i in range(6)]
    col_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}

    def check(player):
        for row in range(6):
            for col in range(7):
                if col + 3 < 7 and all(
                    matrix[row][col + i] == player for i in range(4)
                ):
                    return player
                if row + 3 < 6 and all(
                    matrix[row + i][col] == player for i in range(4)
                ):
                    return player
                if (
                    row + 3 < 6
                    and col + 3 < 7
                    and all(matrix[row + i][col + i] == player for i in range(4))
                ):
                    return player
                if (
                    row + 3 < 6
                    and col - 3 >= 0
                    and all(matrix[row + i][col - i] == player for i in range(4))
                ):
                    return player
        return None

    for move in pieces_position_list:
        col_str, player = move.split("_")
        col = col_map[col_str]

        for row in range(5, -1, -1):
            if matrix[row][col] == 0:
                matrix[row][col] = player
                break
        winner = check(player)
        if winner:
            return winner
    return "Draw"


# 6kyu Build Tower
def tower_builder(n_floors):
    res = []
    w = 2 * n_floors - 1
    for i in range(1, n_floors + 1):
        star = "*" * (2 * i - 1)
        space = " " * ((w - len(star)) // 2)
        res.append(f"{space}{star}{space}")
    return res


# 6kyu Unique In Order
def unique_in_order(sequence):
    res = []
    list_srting = [char for char in sequence]
    for i in range(len(list_srting)):
        if i == 0:
            res.append(list_srting[i])
        elif list_srting[i] != list_srting[i - 1]:
            res.append(list_srting[i])
    return res


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
