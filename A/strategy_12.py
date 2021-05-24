"""
Optunaでチューニングできるようにした
マップパラメータの予測を試す
M=2のときの経路を二分する箇所を特定し、左右2群の差を見てみる
"""

import sys
import argparse
import time
import logging
import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge
from scipy.stats import ttest_ind


input = sys.stdin.readline
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


def smoothing(M, edges, window_func=np.ones, window_size=7, param_D=100):
    assert window_size % 2 == 1
    M1M = (M - 1) * M
    x = edges
    window = window_func(window_size)
    window /= np.trapz(window)

    for i in range(M):
        # smooth horizontal lane
        l = i * (M - 1)
        r = l + M - 1
        horizontal_in = np.pad(x[l:r], (window_size - 1, window_size - 1), "edge")
        x[l:r] = np.convolve(horizontal_in, window, mode="same")[
            window_size - 1 : -window_size + 1
        ]

        # smooth vertical lane
        l = M1M + i
        r = M1M * 2
        vertical_in = np.pad(x[l:r:M], (window_size - 1, window_size - 1), "edge")
        x[l:r:M] = np.convolve(vertical_in, window, mode="same")[
            window_size - 1 : -window_size + 1
        ]

    # min-max正規化
    x = (x - x.min()) / (x.max() - x.min())
    # [0, 1] -> [1000, 9000]
    x = x * 8000 + 1000
    return x


def pivot_smoothing(M, edges, pivot, window_func=np.ones, window_size=7, param_D=100):
    assert window_size % 2 == 1
    M1M = (M - 1) * M
    x = edges
    window = window_func(window_size)
    window /= np.trapz(window)

    for i in range(M):
        # smooth horizontal lane
        l = i * (M - 1)
        r = l + M - 1
        p = i * (M - 1) + pivot[i]
        if p > l:
            horizontal_in = np.pad(x[l:p], (window_size - 1, window_size - 1), "edge")
            x[l:p] = np.convolve(horizontal_in, window, mode="same")[
                window_size - 1 : -window_size + 1
            ]
        horizontal_in = np.pad(x[p:r], (window_size - 1, window_size - 1), "edge")
        x[p:r] = np.convolve(horizontal_in, window, mode="same")[
            window_size - 1 : -window_size + 1
        ]

        # smooth vertical lane
        l = M1M + i
        r = M1M * 2
        p = M1M + pivot[M + i] * M + i
        if p > l:
            vertical_in = np.pad(x[l:p:M], (window_size - 1, window_size - 1), "edge")
            x[l:p:M] = np.convolve(vertical_in, window, mode="same")[
                window_size - 1 : -window_size + 1
            ]
        vertical_in = np.pad(x[p:r:M], (window_size - 1, window_size - 1), "edge")
        x[p:r:M] = np.convolve(vertical_in, window, mode="same")[
            window_size - 1 : -window_size + 1
        ]

    # min-max正規化
    x = (x - x.min()) / (x.max() - x.min())
    # [0, 1] -> [1000, 9000]
    x = x * 8000 + 1000
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
    plt.close()
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
    plt.close()
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
    plt.close()
    logger.info(f"Save heatmap from graph.")


def export_pivot(M, pivot, outfile: str, param_M=0):
    import matplotlib.pyplot as plt

    MM = M * M
    M1M = (M - 1) * M
    y = np.full((M * 2, M * 2), np.nan, dtype=np.float32)
    for i in range(M):
        for j in range(M - 1):
            # horizontal lane
            if pivot[i] == j:
                y[i * 2, j * 2 + 1] = 2.0
            else:
                y[i * 2, j * 2 + 1] = 1.0

    for i in range(M - 1):
        for j in range(M):
            if pivot[M + j] == i:
                y[i * 2 + 1, j * 2] = -2.0
            else:
                y[i * 2 + 1, j * 2] = -1.0
    plt.matshow(y, cmap="BrBG")
    plt.colorbar()
    plt.savefig(outfile)
    plt.close()
    logger.info(f"Save pivot.")


def estimate_map(M, edges, pivot, alpha=0.01):
    M1M = (M - 1) * M
    x = edges
    pvalues = []
    for i in range(M):
        # horizontal lane
        l = i * (M - 1)
        r = l + M - 1
        p = i * (M - 1) + pivot[i]
        if l + 3 * 1 < p < r - 3 * 1:
            statistic, pvalue = ttest_ind(x[l:p], x[p:r], equal_var=True)
            pvalues.append(pvalue)
        # vertical lane
        l = M1M + i
        r = M1M * 2
        p = M1M + pivot[M + i] * M + i
        if l + 3 * M < p < r - 3 * M:
            statistic, pvalue = ttest_ind(x[l:p:M], x[p:r:M], equal_var=True)
            pvalues.append(pvalue)
    if max(pvalues) < alpha:
        return 1
    return 0


def get_pivot(M, edges):
    M1M = (M - 1) * M
    pivot = np.zeros(M + M, dtype=np.int32)
    for i in range(M):
        # smooth horizontal lane
        l = i * (M - 1)
        r = l + M - 1
        imax = 0
        dmax = -1
        lane = np.pad(edges[l:r], (1, 1), "edge")
        for m in range(0, 29):
            lm, rm = lane[: m + 1].mean(), lane[m + 1 :].mean()
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
        lane = np.pad(edges[l:r:M], (1, 1), "edge")
        for m in range(0, 29):
            lm, rm = lane[: m + 1].mean(), lane[m + 1 :].mean()
            d = abs(rm - lm)
            if d > dmax:
                imax = m
                dmax = d
        pivot[M + i] = imax
    return pivot


def solve(
    N=1000,
    M=30,
    n_warmup=100,
    batch_warmup=25,
    batch_cooldown=50,
    window_func="rectangle",
    window_size=11,
    estimate_time=300,
):
    # parameters
    N_WARMUP = n_warmup
    BATCH_WARMUP = batch_warmup
    BATCH_COOLDOWN = batch_cooldown
    ESTIMATE_TIME = estimate_time
    WINDOW_FUNC = {
        "rectangle": np.ones,
        "triangle": np.bartlett,
        "hanning": np.hanning,
        "hamming": np.hamming,
        "blackman": np.blackman,
    }[window_func]
    WINDOW_SIZE = window_size
    M1M = (M - 1) * M

    # 線形回帰モデル
    regressor = Ridge(solver="lsqr", tol=1e-3)
    X = np.empty((N + 1, M1M * 2), dtype=np.float32)
    Y = np.empty(N + 1, dtype=np.float32)
    # レーンごとの境界index (M=1のマップに対してy=[0, M)の整数が入る)
    lane_pivot = np.zeros(M + M, dtype=np.int32)
    G = init_graph(M)
    X[0, :] = 1
    Y[0] = 5000.0 * M1M * 2
    edges = None
    # マップパラメータ (M={0, 1})
    param_M = 0
    param_D = 100

    # Warmup phase
    for i in range(N_WARMUP):
        s, t = read_query(M)

        cost, prev = dijkstra(G, directed=False, indices=s, return_predecessors=True)
        best_route, best_dirs = trace_route(M, prev, s, t)

        write_path(best_dirs)
        actual_cost = read_cost()
        X[i + 1] = route2x(M, best_route, best_dirs)
        Y[i + 1] = actual_cost

        if (i + 1) % BATCH_WARMUP == 0:
            x = X[: i + 1]
            y = Y[: i + 1]
            regressor.fit(x, y)
            edges = smoothing(M, regressor.coef_, WINDOW_FUNC, WINDOW_SIZE, param_D)
            G = update_graph_from(M, G, edges)

    # Cooldown phase
    for i in range(N_WARMUP, N):
        s, t = read_query(M)

        cost, prev = dijkstra(G, directed=False, indices=s, return_predecessors=True)
        best_route, best_dirs = trace_route(M, prev, s, t)

        write_path(best_dirs)
        actual_cost = read_cost()
        X[i + 1] = route2x(M, best_route, best_dirs)
        Y[i + 1] = actual_cost

        if (i + 1) % BATCH_COOLDOWN == 0:
            x = X[: i + 1]
            y = Y[: i + 1]
            regressor.fit(x, y)
            if param_M == 1:
                edges = pivot_smoothing(
                    M, regressor.coef_, lane_pivot, WINDOW_FUNC, WINDOW_SIZE, param_D
                )
            else:
                edges = smoothing(M, regressor.coef_, WINDOW_FUNC, WINDOW_SIZE, param_D)
            G = update_graph_from(M, G, edges)

        if (i + 1) == ESTIMATE_TIME:
            lane_pivot = get_pivot(M, edges)
            param_M = estimate_map(M, edges, lane_pivot)


def parse_arguments():
    parser = argparse.ArgumentParser(description="AHC003 solver.")
    parser.add_argument(
        "--n_warmup",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--batch_warmup",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--batch_cooldown",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--window_func",
        type=str,
        default="rectangle",
        choices=[
            "rectangle",
            "triangle",
            "hanning",
        ],
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=11,
    )
    parser.add_argument(
        "--estimate_time",
        type=int,
        default=300,
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
