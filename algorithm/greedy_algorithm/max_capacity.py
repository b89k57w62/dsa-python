def max_capacity(height: list[int]) -> int:
    """
    max capacity of water that can be stored between two lines
    time complexity: O(n)
    space complexity: O(1)
    """
    left_idx, right_idx = 0, len(height) - 1
    max_capacity = 0
    while left_idx <= right_idx:
        current_capacity = (right_idx - left_idx) * min(
            height[left_idx], height[right_idx]
        )
        max_capacity = max(max_capacity, current_capacity)
        if height[left_idx] < height[right_idx]:
            left_idx += 1
        else:
            right_idx -= 1
    return max_capacity
