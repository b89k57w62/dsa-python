from utils.tree_node import TreeNode


class AVLTree:
    def __init__(self):
        self.root = None

    def get_height(self, node):
        return node.height if node else -1

    def update_height(self, node):
        node.height = max(self.get_height(node.left), self.get_height(node.right)) + 1

    def balance_factor(self, node):
        return self.get_height(node.left) - self.get_height(node.right) if node else 0

    def rotate_right(self, node):
        child = node.left
        child_right = child.right
        child.right = node
        node.left = child_right
        self.update_height(node)
        self.update_height(child)
        return child

    def rotate_left(self, node):
        child = node.right
        child_left = child.left
        child.left = node
        node.right = child_left
        self.update_height(node)
        self.update_height(child)
        return child

    def rotate(self, node):
        balance_factor = self.balance_factor(node)
        if balance_factor > 1:
            if self.balance_factor(node.left) >= 0:
                return self.rotate_right(node)
            else:
                node.left = self.rotate_left(node.left)
                return self.rotate_right(node)
        elif balance_factor < -1:
            if self.balance_factor(node.right) <= 0:
                return self.rotate_left(node)
            else:
                node.right = self.rotate_right(node.right)
                return self.rotate_left(node)
        return node

    def insert(self, val):
        self.root = self.insert_helper(self.root, val)

    def insert_helper(self, node, val):
        if node is None:
            return TreeNode(val)
        if val < node.val:
            node.lefr = self.insert_helper(node.left, val)
        elif val > node.val:
            node.right = self.insert_helper(node.right, val)
        else:
            return node
        self.update_height(node)
        return self.rotate(node)
