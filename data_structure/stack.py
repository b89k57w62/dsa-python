from data_structure.basic_array_list.linked_list import ListNode


class LinkedListStack:
    """base on linked list to implement stack"""

    def __init__(self) -> None:
        self._peek: ListNode | None = None
        self._size = 0

    def size(self) -> int:
        return self._size

    def is_empty(self) -> bool:
        return self._size == 0

    def push(self, val: int):
        node = ListNode(val)
        node.next = self._peek
        self._peek = node
        self._size += 1

    def pop(self) -> int:
        if self.is_empty():
            raise IndexError("Stack is empty")
        val = self._peek.val
        self._peek = self._peek.next
        self._size -= 1
        return val

    def peek(self) -> int:
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self._peek.val

    def to_list(self) -> list[int]:
        node = self._peek
        res = []
        while node:
            res.append(node.val)
            node = node.next
        res.reverse()
        return res


class ArrayStack:
    """base on array to implement stack"""

    def __init__(self) -> None:
        self._stack = []

    def size(self) -> int:
        return len(self._stack)

    def is_empty(self) -> bool:
        return len(self._stack) == 0

    def push(self, val: int):
        self._stack.append(val)

    def pop(self) -> int:
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self._stack.pop()

    def peek(self) -> int:
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self._stack[-1]

    def to_list(self) -> list[int]:
        return self._stack
