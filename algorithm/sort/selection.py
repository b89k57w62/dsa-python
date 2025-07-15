def selection_sort(array):
    """
    Time Complexity: O(n^2)
    Space Complexity: O(1)
    """
    array_length = len(array)
    for current_position in range(array_length - 1):
        minimum_value_index = current_position
        for search_position in range(current_position + 1, array_length):
            if array[search_position] < array[minimum_value_index]:
                minimum_value_index = search_position
        array[current_position], array[minimum_value_index] = (
            array[minimum_value_index],
            array[current_position],
        )
    return array


# 非穩定排序，不保證相同元素的相對位置
# 適用於小規模資料
