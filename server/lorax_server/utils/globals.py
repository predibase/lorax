SPECULATION_NUM = 0


def get_speculation_num() -> int:
    global SPECULATION_NUM
    return SPECULATION_NUM


def set_speculation_num(speculate: int):
    global SPECULATION_NUM
    SPECULATION_NUM = speculate
