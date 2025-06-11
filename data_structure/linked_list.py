class ListNode:
    def __init__(self, val: int, next=None):
        self.val = val
        self.next: ListNode | None = next


def insert(node_1: ListNode, node_2: ListNode):
    temp_node = node_1.next
    node_1.next = node_2
    node_2.next = temp_node


def remove(node: ListNode):
    temp = node.next
    new_next_node = temp.next
    node.next = new_next_node


def access(head: ListNode, index: int):
    for _ in range(index):
        head = head.next
    return head
