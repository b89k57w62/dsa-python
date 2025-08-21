class Item:
    def __init__(self, weight: int, value: int):
        self.weight = weight
        self.value = value


def fractional_knapsack(wgt: list[int], val: list[int], capacity: int):
    items = [Item(w, v) for w, v in zip(wgt, val)]
    items.sort(key=lambda item: item.value / item.weight, reverse=True)
    total_value = 0
    for item in items:
        if item.weight <= capacity:
            capacity -= item.weight
            total_value += item.value
        else:
            total_value += (item.value / item.weight) * capacity
            break
    return total_value
