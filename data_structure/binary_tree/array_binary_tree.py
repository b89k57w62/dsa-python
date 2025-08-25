class ArrayBinaryTree:
    def __init__(self, arr: list[int]):
        self._tree = list(arr)

    def size(self):
        return len(self._tree)

    def _idx_to_val(self, idx: int):
        if idx < 0 or idx >= self.size():
            return None
        return self._tree[idx]

    def left_child_idx(self, idx: int):
        return 2 * idx + 1

    def right_child_idx(self, idx: int):
        return 2 * idx + 2

    def parent_idx(self, idx: int):
        return (idx - 1) // 2

    def _dfs(self, idx: int, order: str):
        if self._idx_to_val(idx) is None:
            return
        if order == "pre":
            self.res.append(self._idx_to_val(idx))
        self._dfs(self.left_child_idx(idx), order)

        if order == "in":
            self.res.append(self._idx_to_val(idx))
        self._dfs(self.right_child_idx(idx), order)

        if order == "post":
            self.res.append(self._idx_to_val(idx))

    def pre_order(self):
        self.res = []
        self._dfs(0, "pre")
        return self.res

    def in_order(self):
        self.res = []
        self._dfs(0, "in")
        return self.res

    def post_order(self):
        self.res = []
        self._dfs(0, "post")
        return self.res

    def level_order(self):
        self.res = []
        for i in range(self.size()):
            if self._idx_to_val(i) is not None:
                self.res.append(self._idx_to_val(i))
        return self.res
