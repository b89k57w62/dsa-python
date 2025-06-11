from data_structure.basic_array_list.linked_list import ListNode


class LinkedListQueue:
    """base on linked list to implement queue"""

    def __init__(self) -> None:
        self._front: ListNode | None = None
        self._rear: ListNode | None = None
        self._size = 0

    def size(self) -> int:
        return self._size

    def is_empty(self) -> bool:
        return self._size == 0

    def push(self, val: int):
        node = ListNode(val)
        if self._front is None:
            self._front = node
            self._rear = node
        else:
            self._rear.next = node
            self._rear = node
        self._size += 1

    def pop(self) -> int:
        if self.is_empty():
            raise ImportError("Queue is empty")
        val = self._front.val
        self._front = self._front.next
        self._size -= 1
        return val

    def peek(self) -> int:
        if self.is_empty():
            raise ImportError("Queue is empty")
        return self._front.val

    def to_list(self) -> list[int]:
        node = self._front
        res = []
        while node:
            res.append(node.val)
            node = node.next
        return res


class ArrayQueue:
    """base on array to implement queue"""

    def __init__(self, size: int) -> None:
        self._front = 0
        self._nums = [0] * size
        self._size = 0

    def size(self) -> int:
        return self._size

    def is_empty(self) -> bool:
        return self._size == 0

    def capacity(self) -> int:
        return len(self._nums)

    def push(self, val: int):
        if self._size == self.capacity():
            raise ImportError("Queue is full")
        rear = (self._front + self._size) % self.capacity()
        self._nums[rear] = val
        self._size += 1

    def pop(self):
        if self.is_empty():
            raise ImportError("Queue is empty")
        val = self._nums[self._front]
        self._front = (self._front + 1) % self.capacity()
        self._size -= 1
        return val

    def peek(self) -> int:
        if self.is_empty():
            raise ImportError("Queue is empty")
        return self._nums[self._front]

    def to_list(self) -> list[int]:
        res = [0] * self.size()
        j = self._front
        for i in range(self.size()):
            res[i] = self._nums[j % self.capacity()]
            j += 1
        return res
