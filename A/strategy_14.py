"""
窓関数を決める
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


def read_query(M):
    si, sj, ti, tj = map(int, input().strip().split())
    return si * M + sj, ti * M + tj


def read_cost():
    return int(input().strip())


def trace_route(M, prev, start, end):
    # prev: (900, )
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


def init_graph(M):
    MM = M * M
    size = M * (M - 1) * 2
    data = np.full(size, 5000, dtype=np.int32)
    s = np.empty(size, dtype=np.int32)
    t = np.empty(size, dtype=np.int32)
    k = 0
    # horizontal lane
    for i in range(M):
        for j in range(M - 1):
            s[k] = i * M + j
            t[k] = i * M + j + 1
            k += 1
    # vertical lane
    for i in range(M - 1):
        for j in range(M):
            s[k] = i * M + j
            t[k] = (i + 1) * M + j
            k += 1
    G = csr_matrix((data, (s, t)), shape=(MM, MM))
    return G


def smoothing(M, edges, window_func, window_size):
    assert window_size % 2 == 1
    M1M = (M - 1) * M
    x = edges
    x = (x - x.min()) / (x.max() - x.min())
    x = x * 8000 + 1000
    window = window_func(window_size)
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


def update_graph_from(M, G, edges):
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


def write_path(dirs):
    DIRECTIONS_STR = "DRLU"
    print("".join([DIRECTIONS_STR[i] for i in dirs]), flush=True)


def update_graph(M, graph, route, dirs, update_func, w=0):
    x = [0 for k in range(len(dirs))]
    c = [0 for k in range(len(dirs))]
    for k in range(len(dirs)):
        si, sj = divmod(route[k], M)
        ti, tj = divmod(route[k + 1], M)
        t = route[k + 1]
        d = dirs[k]
        if d == 3:
            # TOP (vertical)
            for i in range(ti - w, ti + w + 1):
                if 0 <= i < M:
                    x[k] += graph[i * M + tj][0]
                    c[k] += 1
        elif d == 2:
            # LEFT (horizontal)
            for j in range(tj - w, tj + w + 1):
                if 0 <= j < M:
                    x[k] += graph[ti * M + j][1]
                    c[k] += 1
        elif d == 1:
            # RIGHT (horizontal)
            for j in range(sj - w, sj + w + 1):
                if 0 <= j < M:
                    x[k] += graph[si * M + j][1]
                    c[k] += 1
        elif d == 0:
            # DOWN (vertical)
            for i in range(si - w, si + w + 1):
                if 0 <= i < M:
                    x[k] += graph[i * M + sj][0]
                    c[k] += 1
    for k in range(len(dirs)):
        s = route[k]
        t = route[k + 1]
        d = dirs[k]
        graph[s][d] = update_func(x[k] // c[k])
        graph[t][3 - d] = update_func(x[k] // c[k])


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


def solve(N=1000, M=30, window_func="hanning", window_size=19):
    WINDOW_FUNC = {
        "rectangle": np.ones,
        "triangle": np.bartlett,
        "hanning": np.hanning,
        "hamming": np.hamming,
        "blackman": np.blackman,
    }[window_func]
    WINDOW_SIZE = window_size
    BATCH = 50
    M1M = (M - 1) * M

    regressor = Ridge(solver="lsqr", tol=1e-3)
    X = np.empty((N + 1, M1M * 2), dtype=np.float32)
    Y = np.empty(N + 1, dtype=np.float32)
    G = init_graph(M)
    X[0, :] = 1
    Y[0] = (
        regressor.coef_.sum()
        if regressor and hasattr(regressor, "coef_")
        else 5000.0 * M1M * 2
    )

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
        edges = smoothing(M, regressor.coef_, WINDOW_FUNC, WINDOW_SIZE)
        G = update_graph_from(M, G, edges)


def parse_arguments():
    parser = argparse.ArgumentParser(description="AHC003 solver.")
    parser.add_argument(
        "--window_func",
        type=str,
        default="hanning",
        choices=[
            "rectangle",
            "triangle",
            "hanning",
        ],
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=19,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    logging.basicConfig(level=logging.INFO)
    s = time.time()
    solve(**vars(args))
    t = time.time()
    logger.info(t - s)
    logger.info(args)
