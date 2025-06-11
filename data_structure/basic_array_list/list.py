class MyList:
    def __init__(self):
        self._capacity = 10
        self._size = 0
        self._arr = [0] * self._capacity
        self._extend_ratio = 2

    def size(self):
        return self._size

    def capacity(self):
        return self._capacity

    def get(self, index: int):
        if index < 0 or index >= self._size:
            raise IndexError("Index out of range")
        return self._arr[index]

    def set(self, index: int, num: int):
        if index < 0 or index >= self._size:
            raise IndexError("Index out of range")
        self._arr[index] = num

    def add(self, num: int):
        if self._size == self._capacity:
            self._extend_capacity()
        self._arr[self._size] = num
        self._size += 1

    def extend_capacity(self):
        self._arr = self._arr + [0] * self.capacity() * (self._extend_ratio - 1)
        self._capacity = len(self._arr)
