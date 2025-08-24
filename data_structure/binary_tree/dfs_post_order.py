from utils.tree_node import TreeNode


def dfs_post_order(root: TreeNode | None):
    if not root:
        return []
    return dfs_post_order(root.left) + dfs_post_order(root.right) + [root.val]
