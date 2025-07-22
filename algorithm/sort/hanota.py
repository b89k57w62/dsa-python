def move(source, target):
    temp = source.pop()
    target.append(temp)


def dfs(n: int, source: list, target: list, temp: list):
    if n == 1:
        move(source, target)
        return
    dfs(n - 1, source, temp, target)
    move(source, target)
    dfs(n - 1, temp, target, source)


def hanota(n: int, source: list, target: list, temp: list):
    dfs(n, source, target, temp)
    return target
