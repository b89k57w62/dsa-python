class ListNode:
    def __init__(self, val):
        """
        節點類別的建構函式。

        Args:
            val: 節點儲存的值。
        """
        self.val = val  # 節點的值
        self.next = None  # 指向下一個節點的指標 (初始為 None)
        self.prev = None  # 指向上一個節點的指標 (初始為 None)


class LinkedListDeque:
    def __init__(self):
        """
        雙向鏈結串列佇列的建構函式。
        """
        self._front = None  # 佇列前端的節點 (初始為 None)
        self._back = None  # 佇列後端的節點 (初始為 None)
        self._size = 0  # 佇列的大小 (初始為 0)

    def size(self):
        """
        回傳佇列的大小。

        Returns:
            int: 佇列中元素的數量。
        """
        return self._size

    def is_empty(self):
        """
        檢查佇列是否為空。

        Returns:
            bool: 如果佇列為空，則為 True，否則為 False。
        """
        return self._size == 0

    def push(self, val, is_front: bool):
        """
        將一個元素推入佇列的前端或後端。

        Args:
            val: 要推入佇列的值。
            is_front: 如果為 True，則推入前端，否則推入後端。
        """
        node = ListNode(val)  # 建立一個新的節點
        if self.is_empty():  # 如果佇列為空
            self._front = self._back = node  # 前端和後端都指向這個新的節點
        elif is_front:  # 如果要推入前端
            node.next = self._front  # 新節點的 next 指向原本的前端節點
            self._front.prev = node  # 原本的前端節點的 prev 指向新節點
            self._front = node  # 更新前端為新節點
        else:  # 如果要推入後端
            node.prev = self._back  # 新節點的 prev 指向原本的後端節點
            self._back.next = node  # 原本的後端節點的 next 指向新節點
            self._back = node  # 更新後端為新節點
        self._size += 1  # 佇列大小加一

    def push_front(self, val):
        """
        將一個元素推入佇列的前端。

        Args:
            val: 要推入佇列的值。
        """
        self.push(val, True)  # 調用 push 方法，並指定推入前端

    def push_back(self, val):
        """
        將一個元素推入佇列的後端。

        Args:
            val: 要推入佇列的值。
        """
        self.push(val, False)  # 調用 push 方法，並指定推入後端

    def pop(self, is_front: bool):
        """
        從佇列的前端或後端彈出一個元素。

        Args:
            is_front: 如果為 True，則從前端彈出，否則從後端彈出。

        Returns:
            any: 被彈出的元素的值。

        Raises:
            IndexError: 如果佇列為空，則引發 IndexError。
        """
        if self.is_empty():  # 如果佇列為空
            raise IndexError("pop from empty deque")  # 引發錯誤
        if is_front:  # 如果要從前端彈出
            temp = self._front.val  # 暫存前端節點的值
            fnext = self._front.next  # 取得前端節點的下一個節點
            if fnext is not None:  # 如果下一個節點存在
                fnext.prev = None  # 下一個節點的 prev 指向 None
            self._front = fnext  # 更新前端為下一個節點
        else:  # 如果要從後端彈出
            temp = self._back.val  # 暫存後端節點的值
            bprev = self._back.prev  # 取得後端節點的前一個節點
            if bprev is not None:  # 如果前一個節點存在
                bprev.next = None  # 前一個節點的next指向None
            self._back = bprev  # 更新後端為前一個節點
        self._size -= 1  # 佇列大小減一
        return temp  # 回傳彈出的值

    def pop_front(self):
        """
        從佇列的前端彈出一個元素。

        Returns:
            any: 被彈出的元素的值。
        """
        return self.pop(True)  # 調用 pop 方法，並指定從前端彈出

    def pop_back(self):
        """
        從佇列的後端彈出一個元素。

        Returns:
            any: 被彈出的元素的值。
        """
        return self.pop(False)  # 調用 pop 方法，並指定從後端彈出


class ArrayDeque:
    def __init__(self, capacity: int):
        """
        陣列佇列的建構函式。

        Args:
            capacity: 佇列的容量。
        """
        self._nums = [0] * capacity  # 使用固定大小的陣列來儲存元素
        self._size = 0  # 佇列的大小 (初始為 0)
        self._front = 0  # 佇列前端的索引 (初始為 0)

    def capacity(self):
        """
        回傳佇列的容量。

        Returns:
            int: 佇列的容量。
        """
        return len(self._nums)

    def size(self):
        """
        回傳佇列的大小。

        Returns:
            int: 佇列中元素的數量。
        """
        return self._size

    def is_empty(self):
        """
        檢查佇列是否為空。

        Returns:
            bool: 如果佇列為空，則為 True，否則為 False。
        """
        return self.size() == 0

    def index(self, i: int):
        """
        計算循環陣列的索引。

        Args:
            i: 原始索引。

        Returns:
            int: 循環陣列的實際索引。
        """
        return (self.capacity() + i) % self.capacity()

    def push_first(self, num: int):
        """
        將一個元素推入佇列的前端。

        Args:
            num: 要推入佇列的值。

        Raises:
            IndexError: 如果佇列已滿，則引發 IndexError。
        """
        if self.size() == self.capacity():  # 如果佇列已滿
            raise IndexError("push to full deque")  # 引發錯誤
        self._front = self.index(self._front - 1)  # 更新前端索引
        self._nums[self._front] = num  # 將元素放入陣列
        self._size += 1  # 佇列大小加一

    def push_last(self, num: int):
        """
        將一個元素推入佇列的後端。

        Args:
            num: 要推入佇列的值。

        Raises:
            IndexError: 如果佇列已滿，則引發 IndexError。
        """
        if self.size() == self.capacity():  # 如果佇列已滿
            raise IndexError("push to full deque")  # 引發錯誤
        back = self.index(self._front + self.size())  # 計算後端索引
        self._nums[back] = num  # 將元素放入陣列
        self._size += 1  # 佇列大小加一

    def peek_first(self):
        """
        查看佇列前端的元素，但不彈出。

        Returns:
            any: 前端的元素的值。

        Raises:
            IndexError: 如果佇列為空，則引發 IndexError。
        """
        if self.is_empty():  # 如果佇列為空
            raise IndexError("peek from empty deque")  # 引發錯誤
        return self._nums[self._front]  # 回傳前端的元素

    def peek_last(self):
        """
        查看佇列後端的元素，但不彈出。

        Returns:
            any: 後端的元素的值。

        Raises:
            IndexError: 如果佇列為空，則引發 IndexError。
        """
        if self.is_empty():  # 如果佇列為空
            raise IndexError("peek from empty deque")  # 引發錯誤
        return self._nums[self.index(self._front + self.size() - 1)]  # 回傳後端的元素

    def pop_first(self):
        """
        從佇列的前端彈出一個元素。

        Returns:
            any: 被彈出的元素的值。

        Raises:
            IndexError: 如果佇列為空，則引發 IndexError
        """
        peeked_element = self.peek_first()
        self._front = self.index(self._front + 1)  # 更新前端索引
        self._size -= 1  # 佇列大小減一
        return peeked_element  # 返回彈出的元素

    def pop_last(self):
        """
        從佇列的後端彈出一個元素。

        Returns:
            any: 被彈出的元素的值。

        Raises:
            IndexError: 如果佇列為空，則引發 IndexError
        """
        peeked_element = self.peek_last()
        self._size -= 1  # 佇列大小減一
        return peeked_element  # 返回彈出的元素
