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
