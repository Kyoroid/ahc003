import numpy as np


def read(M, infile: str):
    from pathlib import Path
    MM = M * M
    M1M = (M-1) * M
    h = np.empty((M, M-1))
    v = np.empty((M-1, M))
    x = np.empty(M1M * 2)
    with Path(infile).open(mode="r") as f:
        # horizontal lane
        for i in range(M):
            h[i] = np.array(list(map(int, f.readline().split())))
        # vertical lane
        for i in range(M - 1):
            v[i] = np.array(list(map(int, f.readline().split())))
    x[:M1M] = h.flatten()
    x[M1M:] = v.flatten()
    return x


def set_pivot(M, x):
    M1M = (M-1) * M
    pivot = np.zeros(M+M, dtype=np.int32)
    for i in range(M):
        # smooth horizontal lane
        l = i * (M - 1)
        r = l + M - 1
        imax = 0
        dmax = -1
        lane = x[l:r]
        for m in range(0, 29):
            lm, rm = lane[:m].mean(), lane[m:].mean()
            d = abs(rm - lm)
            if d > dmax:
                imax = m
                dmax = d
        pivot[i] = imax
        # smooth vertical lane
        l = M1M + i
        r = M1M * 2
        imax = 0
        dmax = -1
        lane = x[l:r:M]
        for m in range(0, 29):
            lm, rm = lane[:m].mean(), lane[m:].mean()
            d = abs(rm - lm)
            if d > dmax:
                imax = m
                dmax = d
        pivot[M+i] = imax
    return pivot

M = 30
edges = read(M, "../tools/in/0050.txt")
pivot = set_pivot(M, edges)
print(",".join(map(str, pivot)))