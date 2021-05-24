"""
最終提出
Optunaでalpha, 窓関数を微調整

predicted score:
933427781 [930001264, 936854298]

atcual score:
93183978165

https://atcoder.jp/contests/ahc003/submissions/23029585

final score:
2,799,306,751,489

https://atcoder.jp/contests/ahc003/submissions/23031833
"""


import sys
import argparse
import time

input = sys.stdin.readline
import logging
import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge


logger = logging.getLogger(__name__)


######## INPUTS & OUTPUTS #########


def read_query(M):
    si, sj, ti, tj = map(int, input().strip().split())
    return si * M + sj, ti * M + tj


def write_path(dirs):
    DIRECTIONS_STR = "DRLU"
    print("".join([DIRECTIONS_STR[i] for i in dirs]), flush=True)


def read_cost():
    return int(input().strip())


######## UTILITIES #########


def export_input(M, infile: str, outfile: str, cmin=1000, cmax=9000):
    import matplotlib.pyplot as plt
    from pathlib import Path

    MM = M * M
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

    MM = M * M
    M1M = (M - 1) * M
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

    MM = M * M
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


######## SOLVER #########


def init_graph(M, edges):
    # G: csr_matrix (MM, MM)
    # edges: (MM * 2)
    MM = M * M
    s = list()
    t = list()
    # horizontal lane
    for i in range(M):
        for j in range(M - 1):
            s.append(i * M + j)
            t.append(i * M + j + 1)
    # vertical lane
    for i in range(M - 1):
        for j in range(M):
            s.append(i * M + j)
            t.append((i + 1) * M + j)
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


def route2x(M, route, dirs):
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


def smoothing(M, edges, window_size):
    assert window_size % 2 == 1
    M1M = (M - 1) * M
    x = edges
    x = (x - x.min()) / (x.max() - x.min())
    x = x * 8000 + 1000
    window = np.hanning(window_size)
    window /= np.trapz(window)
    for i in range(M):
        # smooth horizontal lane
        l = i * (M - 1)
        r = l + M - 1
        x[l:r] = np.convolve(x[l:r], window, mode="same")

        # smooth vertical lane
        l = M1M + i
        r = M1M * 2
        x[l:r:M] = np.convolve(x[l:r:M], window, mode="same")
    return x


def solve(window_size, alpha, N=1000, M=30):
    BATCH = 50
    M1M = (M - 1) * M

    regressor = Ridge(solver="lsqr", tol=1e-3, alpha=alpha)
    X = np.empty((N + 1, M1M * 2), dtype=np.float32)
    Y = np.empty(N + 1, dtype=np.float32)
    edges = np.full(M1M * 2, 5000, dtype=np.int32)
    G = init_graph(M, edges)
    X[0, :] = 1
    Y[0] = 5000.0 * M1M * 2

    for k in range(N // BATCH):

        for i in range(BATCH):
            s, t = read_query(M)

            cost, prev = dijkstra(
                G, directed=False, indices=s, return_predecessors=True
            )
            best_route, best_dirs = trace_route(M, prev, s, t)

            write_path(best_dirs)
            actual_cost = read_cost()
            X[k * BATCH + i + 1] = route2x(M, best_route, best_dirs)
            Y[k * BATCH + i + 1] = actual_cost

        regressor.fit(X[: (k + 1) * BATCH + 1], Y[: (k + 1) * BATCH + 1])
        edges = smoothing(M, regressor.coef_, window_size)
        G = init_graph(M, edges)


def parse_arguments():
    parser = argparse.ArgumentParser(description="AHC003 solver.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.17,
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=17,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    logging.basicConfig(level=logging.ERROR)
    s = time.time()
    solve(**vars(args))
    t = time.time()
    logger.info(t - s)
    logger.info(args)
