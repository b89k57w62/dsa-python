class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        self.height = 0


class AVLTree:
    def __init__(self):
        pass

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
