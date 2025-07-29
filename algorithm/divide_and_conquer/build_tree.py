class TreeBuilder:
    """
    Time complexity: O(n)
    Space complexity: O(n)
    Binary tree builder that constructs a tree from preorder and inorder traversal arrays.
    Uses divide and conquer approach for efficient tree construction.
    """

    def build_tree(self, preorder, inorder):
        """
        Build a binary tree from preorder and inorder traversal arrays.

        Args:
            preorder: List of values in preorder traversal
            inorder: List of values in inorder traversal

        Returns:
            TreeNode: Root of the constructed binary tree
        """
        if not preorder or not inorder:
            return None

        inorder_map = {val: i for i, val in enumerate(inorder)}
        return self._dfs(preorder, inorder_map, 0, 0, len(inorder) - 1)

    def _dfs(self, preorder, inorder_map, pre_start, in_start, in_end):
        """
        Helper method for recursive tree construction using divide and conquer.

        Args:
            preorder: Preorder traversal array
            inorder_map: Map of value to index in inorder array
            pre_start: Current start index in preorder array
            in_start: Start index in inorder array for current subtree
            in_end: End index in inorder array for current subtree

        Returns:
            TreeNode: Root of the current subtree
        """
        if in_start > in_end:
            return None

        root = TreeNode(preorder[pre_start])
        mid = inorder_map[preorder[pre_start]]
        root.left = self._dfs(preorder, inorder_map, pre_start + 1, in_start, mid - 1)
        root.right = self._dfs(
            preorder, inorder_map, pre_start + 1 + mid - in_start, mid + 1, in_end
        )
        return root


class TreeNode:
    """
    Binary tree node representation.
    """

    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
