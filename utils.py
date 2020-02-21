import functools

def clip_to_range(value, lw=-1, up=1):
    if value > up:
        return up
    if value < lw:
        return lw
    return value


def create_interval(value, delta):
    interval = (value - delta, value + delta)
    return interval

def fold(fun, obs, init):
    return functools.reduce(fun, obs, init)