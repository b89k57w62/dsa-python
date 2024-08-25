# 6kyu Delete occurrences of an element if it occurs more than n times
def delete_nth(order,max_e):
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
   bin_num = format(n, 'b')
   return bin_num.count('1')

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
   for str in text.split(' '):
      if str.isalpha():
         ans.append(str.replace(str[0], "", 1)+str[0]+"ay")
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
   return " ".join([words[::-1] if len(words) >=5 else words for words in array])

