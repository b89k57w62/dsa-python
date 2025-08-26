from utils.tree_node import TreeNode


class BinarySearchTree:
    def __init__(self):
        self.root = None | TreeNode

    def search(self, num):
        current_node = self.root
        while current_node is not None:
            if current_node.val == num:
                return current_node
            elif current_node.val < num:
                current_node = current_node.right
            else:
                current_node = current_node.left
        return current_node

    def insert(self, num):
        if self.root is None:
            self.root = TreeNode(num)
            return
        current_node, previous_node = self.root, None
        while current_node is not None:
            previous_node = current_node
            if current_node.val < num:
                current_node = current_node.right
            else:
                current_node = current_node.left
        node = TreeNode(num)
        if previous_node.val < num:
            previous_node.right = node
        else:
            previous_node.left = node

    def remove(self, num):
        if self.root is None:
            return
        current_node, previous_node = self.root, None
        while current_node is not None:
            if current_node.val == num:
                break
            previous_node = current_node
            if current_node.val < num:
                current_node = current_node.right
            else:
                current_node = current_node.left
        if current_node is None:
            return

        if current_node.left is None and current_node.right is None:
            child = current_node.left or current_node.right
            if previous_node.left == current_node:
                previous_node.left = child
            elif previous_node.right == current_node:
                previous_node.right = child
            else:
                self.root = child
        else:
            tmp = current_node.right
            while tmp.left is not None:
                tmp = tmp.left
            self.remove(tmp.val)
            current_node.val = tmp.val
