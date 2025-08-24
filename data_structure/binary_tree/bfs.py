from collections import deque
from utils.tree_node import TreeNode


def level_order(root: TreeNode | None):
    if not root:
        return []
    que = deque()
    que.append(root)
    res = []
    while que:
        node = que.popleft()
        res.append(node.val)
        if node.left:
            que.append(node.left)
        if node.right:
            que.append(node.right)
    return res
