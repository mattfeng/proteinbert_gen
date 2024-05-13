def print2(*args, tags=tuple(), **kwargs):
    active_tags = {"in block", "inputs"}
    debug = False

    if debug and any(map(lambda t: t in active_tags, tags)):
        print(*args, **kwargs)