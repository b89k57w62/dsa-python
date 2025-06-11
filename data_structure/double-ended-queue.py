class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = ListNode | None
        self.prev = ListNode | None


class LinkedListDeque:
    """
    A double-ended queue implemented using a linked list.
    """

    def __init__(self):
        self._front = ListNode | None
        self._rear = ListNode | None
        self._size = 0

    def size(self):
        return self._size

    def is_empty(self):
        return self._size == 0

    def push(self, val, is_front: bool):
        node = ListNode(val)
        if self._front is None:
            self._front = self._rear = node
        elif is_front:
            node.next = self._front
            self._front.prev = node
            self._front = node
        else:
            node.prev = self._rear
            self._rear.next = node
            self._rear = node
        self._size += 1

    def push_first(self, val):
        self.push(val, is_front=True)

    def push_last(self, val):
        self.push(val, is_front=False)

    def pop(self, is_front: bool):
        if self.is_empty():
            raise IndexError("Deque is empty")
        if is_front:
            val = self._front.val
            fnext = self._front.next
            if fnext is not None:
                fnext.prev = None
                self._front.next = None
            self._front = fnext
        else:
            val = self._rear.val
            rprev = self._rear.prev
            if rprev is not None:
                rprev.next = None
                self._rear.prev = None
            self._rear = rprev
        self._size -= 1
        return val

    def pop_first(self):
        return self.pop(is_front=True)

    def pop_last(self):
        return self.pop(is_front=False)

    def peek_first(self):
        if self.is_empty():
            raise IndexError("Deque is empty")
        return self._front.val

    def peek_last(self):
        if self.is_empty():
            raise IndexError("Deque is empty")
        return self._rear.val

    def to_list(self):
        node = self._front
        res = [0] * self.size()
        for i in range(self.size()):
            res[i] = node.val
            node = node.next
        return res