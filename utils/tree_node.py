class TreeNode:
    """
    Binary tree node representation.
    """

    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class AVLTreeNode:
    def __init__(self, val: int):
        self.val = val
        self.left = AVLTreeNode | None
        self.right = AVLTreeNode | None
        self.height = 0
