def digit(num, place_value):
    return (num // place_value) % 10


def counting_sort_digit(arr, place_value):
    digit_counter = [0] * 10
    array_length = len(arr)
    for index in range(array_length):
        # to get the digit at the current place value
        current_digit = digit(arr[index], place_value)
        digit_counter[current_digit] += 1

    # to get the number of elements less than or equal to i
    for digit_index in range(1, 10):
        digit_counter[digit_index] += digit_counter[digit_index - 1]

    sorted_array = [0] * array_length
    for index in range(array_length - 1, -1, -1):
        current_digit = digit(arr[index], place_value)
        position = digit_counter[current_digit] - 1
        sorted_array[position] = arr[index]
        digit_counter[current_digit] -= 1
    for index in range(array_length):
        arr[index] = sorted_array[index]


def radix_sort(arr):
    max_value = max(arr)
    place_value = 1
    while place_value <= max_value:
        counting_sort_digit(arr, place_value)
        place_value *= 10
