from utils.tree_node import TreeNode


class BinarySearchTree:
    def __init__(self):
        self.root = None

    def search(self, num):
        cur = self.root
        while cur is not None:
            if cur.val < num:
                cur = cur.right
            elif cur.val > num:
                cur = cur.left
            else:
                break
        return cur

    def insert(self, num):
        if self.root is None:
            self.root = TreeNode(num)
            return

        cur, pre = self.root, None
        while cur is not None:
            if cur.val == num:
                return
            pre = cur

            if cur.val < num:
                cur = cur.right
            else:
                cur = cur.left
        if pre.val < num:
            pre.right = TreeNode(num)
        else:
            pre.left = TreeNode(num)

    def remove(self, num):
        if self.root is None:
            return

        cur, pre = self.root, None

        while cur is not None:
            if cur.val == num:
                break
            pre = cur
            if cur.val < num:
                cur = cur.right
            else:
                cur = cur.left
        if cur is None:
            return

        if cur.left is None or cur.right is None:
            child = cur.left or cur.right
            if cur != self.root:
                if pre.left == cur:
                    pre.left = child
                else:
                    pre.right = child
            else:
                self.root = child
        else:
            temp = cur.right
            while temp.left is not None:
                temp = temp.left
            self.remove(temp.val)
            cur.val = temp.val
