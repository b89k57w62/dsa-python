class Pair:
    def __init__(self, key: int, val: int):
        self.key = key
        self.val = val


class HashMapChaining:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._size = 0
        self.buckets = [[] for _ in range(self.capacity)]
        self.extend_ratio = 2
        self.extend_threshold = 0.75

    def hash_func(self, key: int):
        return key % self.capacity

    def load_factor(self):
        return self._size / self.capacity

    def get(self, key: int):
        idx = self.hash_func(key)
        bucket = self.buckets[idx]
        for pair in bucket:
            if pair.key == key:
                return pair.val
        return None

    def put(self, key: int, val: int):
        if self.load_factor() > self.extend_threshold:
            self.extend_capacity()
        idx = self.hash_func(key)
        bucket = self.buckets[idx]
        for pair in bucket:
            if pair.key == key:
                pair.val = val
                return
        pair = Pair(key, val)
        bucket.append(pair)
        self._size += 1

    def extend_capacity(self):
        self.capacity *= self.extend_ratio
        temp_buckets = [[] for _ in range(self.capacity)]
        self._size = 0
        for bucket in self.buckets:
            for pair in bucket:
                self.put(pair.key, pair.val)

    def remove(self, key: int):
        if self._size == 0:
            raise IndexError("HashMap is empty")
        idx = self.hash_func(key)
        bucket = self.buckets[idx]
        for pair in bucket:
            if pair.key == key:
                bucket.remove(pair)
                self._size -= 1
                break
