def IoU(l1, l2):

    t1 = tuple(l1)
    t2 = tuple(l2)

    s1 = set(t1)
    s2 = set(t2)

    return len(s1.intersection(s2)) / len(s1.union(s2))

def OverlapCoefficient(l1, l2):

    t1 = tuple(l1)
    t2 = tuple(l2)

    s1 = set(t1)
    s2 = set(t2)

    if len(s1) ==0 or len(s2) == 0:
        return 0

    return len(s1.intersection(s2)) / min(len(s1), len(s2))