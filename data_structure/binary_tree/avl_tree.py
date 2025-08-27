from utils.tree_node import AVLTreeNode


class AVLTree:
    def __init__(self):
        self.root = None

    def height(self, node: AVLTreeNode | None) -> int:
        return node.height if node else -1

    def update_height(self, node: AVLTreeNode | None) -> None:
        node.height = max(self.height(node.left), self.height(node.right)) + 1

    def balance_factor(self, node: AVLTreeNode | None):
        if node is None:
            return 0
        return self.height(node.left) - self.height(node.right)

    def right_rotate(self, node: AVLTreeNode) -> AVLTreeNode:
        child = node.left
        grand_child = child.right
        child.right = node
        node.left = grand_child
        self.update_height(node)
        self.update_height(child)
        return child

    def left_rotate(self, node: AVLTreeNode) -> AVLTreeNode:
        child = node.right
        grand_child = child.left
        child.left = node
        node.right = grand_child
        self.update_height(node)
        self.update_height(child)
        return child

    def rotate(self, node: AVLTreeNode) -> AVLTreeNode:
        balance_factor = self.balance_factor(node)
        if balance_factor > 1:
            if self.balance_factor(node.left) >= 0:
                return self.right_rotate(node)
            else:
                node.left = self.left_rotate(node.left)
                return self.right_rotate(node)
        elif balance_factor < -1:
            if self.balance_factor(node.right) <= 0:
                return self.left_rotate(node)
            else:
                node.right = self.right_rotate(node.right)
                return self.left_rotate(node)
        return node

    def insert(self, val: int) -> None:
        self.root = self._insert(self.root, val)

    def _insert(self, node: AVLTreeNode | None, val: int) -> AVLTreeNode:
        if node is None:
            return AVLTreeNode(val)
        if val < node.val:
            node.left = self._insert(node.left, val)
        else:
            node.right = self._insert(node.right, val)
        self.update_height(node)
        return self.rotate(node)

    def remove(self, val: int) -> None:
        self.root = self._remove(self.root, val)

    def _remove(self, node: AVLTreeNode | None, val: int) -> AVLTreeNode:
        if node is None:
            return None
        if val < node.val:
            node.left = self._remove(node.left, val)
        elif val > node.val:
            node.right = self._remove(node.right, val)
        else:
            if node.left is None or node.right is None:
                child = node.left or node.right
                if child is None:
                    return None
                else:
                    node = child
            else:
                child = node.right
                while child.left is not None:
                    child = child.left
                self.root.val = child.val
                node.right = self._remove(node.right, child.val)
        self.update_height(node)
        return self.rotate(node)
