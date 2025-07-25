def backtrack(state, choices, selected, res):
    if len(state) == len(choices):
        res.append(list(state))
        return

    duplicate = set()
    for i, choice in enumerate(choices):
        if not selected[i] and choice not in duplicate:
            selected[i] = True
            state.append(choice)
            duplicate.add(choice)
            backtrack(state, choices, selected, res)
            state.pop()
            selected[i] = False
