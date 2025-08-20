class Pair:
    def __init__(self, key: int, val: int):
        self.key = key
        self.val = val


class HashMapOpenAddressing:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buckets = [None] * self.capacity
        self._size = 0
        self.extend_ration = 2
        self.extend_threshold = 0.75
        self.tombstone = Pair(-1, -1)

    def hash_func(self, key: int):
        return key % self.capacity

    def load_factor(self):
        return self._size / self.capacity

    def find_bucket(self, key: int):
        idx = self.hash_func(key)
        first_tombstone = -1
        while self.buckets[idx] is not None:
            if self.buckets[idx].key == key:
                if first_tombstone != -1:
                    self.buckets[first_tombstone] = self.buckets[idx]
                    self.buckets[idx] = self.tombstone
                    return first_tombstone
                return idx
            if first_tombstone == -1 and self.buckets[idx] == self.tombstone:
                first_tombstone = idx
            idx = (idx + 1) % self.capacity
        return idx if first_tombstone == -1 else first_tombstone

    def put(self, key: int, val: int):
        if self.load_factor() > self.extend_threshold:
            self.extend_buckets()
        idx = self.find_bucket(key)
        if self.buckets[idx] not in [None, self.tombstone]:
            self.buckets[idx].val = val
            return
        self.buckets[idx] = Pair(key, val)
        self._size += 1

    def extend_buckets(self):
        self.capacity *= self.capacity * self.extend_ration
        temp_buckets = self.buckets
        self.buckets = [None] * self.capacity
        self._size = 0
        for pair in temp_buckets:
            if pair not in [None, self.tombstone]:
                self.put(pair.key, pair.val)

    def get(self, key: int):
        idx = self.find_bucket(key)
        if self.buckets[idx] not in [None, self.tombstone]:
            return self.buckets[idx].val
        return None

    def remove(self, key: int):
        idx = self.find_bucket(key)
        if self.buckets[idx] not in [None, self.tombstone]:
            self.buckets[idx] = self.tombstone
            self._size -= 1
