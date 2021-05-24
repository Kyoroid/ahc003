"""
Numba JIT 対応

predicted score:
95.0% confidence interval: 930963263 [927414557, 934511970]

atcual score:
92927468431

実行時間:
1973 ms

https://atcoder.jp/contests/ahc003/submissions/22973076
"""

import sys

input = sys.stdin.readline
import logging
import numpy as np
from numba import njit
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge


def read_query(M):
    si, sj, ti, tj = map(int, input().strip().split())
    return si * M + sj, ti * M + tj


def write_path(dirs):
    DIRECTIONS_STR = "DRLU"
    print("".join([DIRECTIONS_STR[i] for i in dirs]), flush=True)


def read_cost():
    return int(input().strip())


@njit("i4[:](i4)", cache=True)
def init_sources(M):
    s = np.empty(M * (M - 1) * 2, dtype=np.int32)
    k = 0
    for i in range(M):
        for j in range(M - 1):
            s[k] = i * M + j
            k += 1
    for i in range(M - 1):
        for j in range(M):
            s[k] = i * M + j
            k += 1
    return s


@njit("i4[:](i4)", cache=True)
def init_targets(M):
    t = np.empty(M * (M - 1) * 2, dtype=np.int32)
    k = 0
    for i in range(M):
        for j in range(M - 1):
            t[k] = i * M + j + 1
            k += 1
    for i in range(M - 1):
        for j in range(M):
            t[k] = (i + 1) * M + j
            k += 1
    return t


def init_graph(M, edges):
    MM = M * M
    s = init_sources(M)
    t = init_targets(M)
    G = csr_matrix((edges, (s, t)), shape=(MM, MM))
    return G


def trace_route(M, prev, start, end):
    DIRECTIONS = [-M, -1, 1, M]
    t = end
    route = [t]
    dirs = []
    while t != start:
        s = prev[t]
        route.append(s)
        for i in range(4):
            if s - t == DIRECTIONS[i]:
                dirs.append(i)
        t = s
    return route[::-1], dirs[::-1]


def encode_route(M, route, dirs):
    M1M = (M - 1) * M
    x = np.zeros(M1M * 2, dtype=np.float32)
    for j in range(len(dirs)):
        s = route[j]
        d = dirs[j]
        if d == 3:
            # TOP (vertical)
            x[M1M + s - M] = 1.0
        elif d == 2:
            # LEFT (horizontal)
            si, sj = divmod(s, M)
            x[si * (M - 1) + sj - 1] = 1.0
        elif d == 1:
            # RIGHT (horizontal)
            si, sj = divmod(s, M)
            x[si * (M - 1) + sj] = 1.0
        elif d == 0:
            # DOWN (vertical)
            x[M1M + s] = 1.0
    return x


@njit("f4[:](i4, f4[:], i4)", cache=True)
def smoothing(M, edges, window_size=7):
    M1M = (M - 1) * M
    x = (edges - edges.min()) / (edges.max() - edges.min())
    x = x * 8000 + 1000
    window = np.ones(window_size)
    window /= np.trapz(window)
    for i in range(M):
        # smooth horizontal lane
        l = i * (M - 1)
        r = l + M - 1
        x[l:r] = np.convolve(x[l:r], window)[
            (window_size - 1) // 2 : -(window_size - 1) // 2
        ]
        # smooth vertical lane
        l = M1M + i
        r = M1M * 2
        x[l:r:M] = np.convolve(x[l:r:M], window)[
            (window_size - 1) // 2 : -(window_size - 1) // 2
        ]
    return x


def export_input(M, infile: str, outfile: str, cmin=1000, cmax=9000):
    import matplotlib.pyplot as plt
    from pathlib import Path

    y = np.full((M * 2, M * 2), np.nan, dtype=np.float32)
    with Path(infile).open(mode="r") as f:
        # horizontal lane
        for i in range(M):
            x = list(map(int, f.readline().split()))
            for j in range(M - 1):
                y[i * 2, j * 2 + 1] = x[j]
        # vertical lane
        for i in range(M - 1):
            x = list(map(int, f.readline().split()))
            for j in range(M):
                y[i * 2 + 1, j * 2] = x[j]
    plt.matshow(y, cmap="bwr")
    plt.colorbar()
    plt.clim(cmin, cmax)
    plt.savefig(outfile)
    logger.info(f"Save heatmap from {infile}.")


def export_edges(M, edges, outfile: str, cmin=1000, cmax=9000):
    import matplotlib.pyplot as plt

    y = np.full((M * 2, M * 2), np.nan, dtype=np.float32)
    k = 0
    for i in range(M):
        for j in range(M - 1):
            # horizontal lane
            y[i * 2, j * 2 + 1] = edges[k]
            k += 1
    for i in range(M - 1):
        for j in range(M):
            # vertical lane
            y[i * 2 + 1, j * 2] = edges[k]
            k += 1
    plt.matshow(y, cmap="bwr")
    plt.colorbar()
    plt.clim(cmin, cmax)
    plt.savefig(outfile)
    logger.info(f"Save heatmap from edges.")


def export_graph(M, G, outfile: str, cmin=1000, cmax=9000):
    import matplotlib.pyplot as plt

    y = np.full((M * 2, M * 2), np.nan, dtype=np.float32)
    # horizontal lane
    for i in range(M):
        for j in range(M - 1):
            s = i * M + j
            t = i * M + j + 1
            y[i * 2, j * 2 + 1] = G[s, t]
    # vertical lane
    for i in range(M - 1):
        for j in range(M):
            s = i * M + j
            t = (i + 1) * M + j
            y[i * 2 + 1, j * 2] = G[s, t]
    plt.matshow(y, cmap="bwr")
    plt.colorbar()
    plt.clim(cmin, cmax)
    plt.savefig(outfile)
    logger.info(f"Save heatmap from graph.")


def solve(N=1000, M=30):
    BATCH = 50
    M1M = (M - 1) * M

    regressor = Ridge(solver="lsqr", tol=1e-3)

    X = np.empty((N + 1, M1M * 2), dtype=np.float32)
    Y = np.empty(N + 1, dtype=np.float32)
    initializer = np.full(M1M * 2, 5000, dtype=np.float32)
    G = init_graph(M, initializer)
    X[0, :] = 1
    Y[0] = M1M * 2 * 5000

    for k in range(N // BATCH):

        for i in range(BATCH):
            s, t = read_query(M)

            cost, prev = dijkstra(
                G, directed=False, indices=s, return_predecessors=True
            )
            best_route, best_dirs = trace_route(M, prev, s, t)

            write_path(best_dirs)
            actual_cost = read_cost()
            X[k * BATCH + i + 1] = encode_route(M, best_route, best_dirs)
            Y[k * BATCH + i + 1] = actual_cost

        regressor.fit(X[: (k + 1) * BATCH + 1], Y[: (k + 1) * BATCH + 1])
        edges = smoothing(M, regressor.coef_, 13)
        G = init_graph(M, edges)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    import time

    s = time.time()
    solve()
    t = time.time()
    logger.info(t - s)
