from utils.tree_node import TreeNode


def dfs_in_order(root: TreeNode | None):
    if not root:
        return []
    return dfs_in_order(root.left) + [root.val] + dfs_in_order(root.right)
