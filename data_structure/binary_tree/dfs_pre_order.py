from utils.tree_node import TreeNode


def dfs_pre_order(root: TreeNode | None):
    if not root:
        return []
    return [root.val] + dfs_pre_order(root.left) + dfs_pre_order(root.right)
