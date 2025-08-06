def min_path_sum_dfs(grid: list[list[int]], row: int, col: int):
    """
    Minimum Path Sum: Brute-force Search
    time complexity: O(2^(m+n))
    space complexity: O(m+n)
    """
    if row == 0 and col == 0:
        return grid[0][0]
    if row < 0 or col < 0:
        return float("inf")
    previous_row_up = min_path_sum_dfs(grid, row - 1, col)
    previous_col_left = min_path_sum_dfs(grid, row, col - 1)
    return grid[row][col] + min(previous_row_up, previous_col_left)


def min_path_sum_dfs_mem(
    grid: list[list[int]], mem: list[list[int]], row: int, col: int
):
    """
    Minimum Path Sum: Brute-force Search with Memoization
    time complexity: O(m*n)
    space complexity: O(m*n)
    """
    if row == 0 and col == 0:
        return grid[0][0]
    if row < 0 or col < 0:
        return float("inf")
    if mem[row][col] != -1:
        return mem[row][col]
    previous_row_up = min_path_sum_dfs_mem(grid, mem, row - 1, col)
    previous_col_left = min_path_sum_dfs_mem(grid, mem, row, col - 1)
    mem[row][col] = grid[row][col] + min(previous_row_up, previous_col_left)
    return mem[row][col]


def min_path_sum_dp(grid: list[list[int]]):
    """
    Minimum Path Sum: Dynamic Programming
    time complexity: O(m*n)
    space complexity: O(m*n)
    """
    rows, cols = len(grid), len(grid[0])
    # initialize the dp table
    dp_table = [[0] * cols for _ in range(rows)]
    dp_table[0][0] = grid[0][0]

    # fill the first row
    for row in range(1, rows):
        dp_table[row][0] = dp_table[row - 1][0] + grid[row][0]

    # fill the first column
    for col in range(1, cols):
        dp_table[0][col] = dp_table[0][col - 1] + grid[0][col]

    # fill the rest of the dp table
    for row in range(1, rows):
        for col in range(1, cols):
            dp_table[row][col] = (
                min(dp_table[row - 1][col], dp_table[row][col - 1]) + grid[row][col]
            )
    return dp_table[rows - 1][cols - 1]
