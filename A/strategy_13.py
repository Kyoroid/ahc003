"""
マップパラメータの予測を試す
2群の差を見たりカイ2乗検定試したりしたけど捨てた
"""

import sys
import argparse
import time
import logging
import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge


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


def smoothing(M, edges, window_size=19):
    assert window_size % 2 == 1
    M1M = (M - 1) * M
    x = edges
    window = np.hanning(window_size)
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


def export_edges(M, edges, outfile: str, cmin=1000, cmax=9000, cmap="bwr"):
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
    plt.matshow(y, cmap=cmap)
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


def export_count(M, count, outfile):
    import matplotlib.pyplot as plt
    plt.hist(count, bins=30, range=(0, 30))
    plt.savefig(outfile)
    plt.close()


def estimate_yh(M, edges):
    M1M = (M - 1) * M
    y = np.zeros(M + M, dtype=np.int32)
    h = np.full(M * 4, 5000, dtype=np.int32)
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
        y[i] = imax
        h[i*2+1] = edges[l+imax:r].mean()
        if imax > 0:
            h[i*2] = edges[l:l+imax].mean()
        else:
            h[i*2] = h[i*2+1]
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
        y[M + i] = imax
        h[M+M+i*2+1] = edges[l+imax*M:r:M].mean()
        if imax > 0:
            h[M+M+i*2] = edges[l:l+imax*M:M].mean()
        else:
            h[M+M+i*2] = h[M+M+i*2+1]
    return y, h


def edges_from_yh(M, y, h):
    M1M = (M - 1) * M
    edges = np.full(M1M*2, 5000, dtype=np.float32)
    for i in range(M):
        l = i * (M - 1)
        r = l + M - 1
        m = l + y[i]
        edges[l:m] = h[i*2]
        edges[m:r] = h[i*2+1]
        l = M1M + i
        r = M1M * 2
        m = l + y[i] * M
        edges[l:m:M] = h[M+M+i*2]
        edges[m:r:M] = h[M+M+i*2+1]
    return edges



def solve(
    batch,
    N=1000,
    M=30,
):
    # parameters
    BATCH = batch
    M1M = (M - 1) * M

    # 線形回帰モデル
    regressor = Ridge(solver="lsqr", tol=1e-3, alpha=0.01)
    
    # 観測値
    X = np.empty((N + 1, M1M * 2), dtype=np.float32)
    Y = np.empty(N + 1, dtype=np.float32)
    X[0, :] = 1
    Y[0] = 5000.0 * M1M * 2

    # マップパラメータ
    param_M = 0
    param_D = 100
    param_H = np.full(M * 4, 5000, dtype=np.float32)
    param_Y = np.zeros(M + M, dtype=np.int32)

    # グラフ
    edges = np.full(M1M*2, 5000, dtype=np.float32)
    G = init_graph(M, edges)

    # Warmup phase
    for i in range(N):

        s, t = read_query(M)

        cost, prev = dijkstra(G, directed=False, indices=s, return_predecessors=True)
        best_route, best_dirs = trace_route(M, prev, s, t)

        write_path(best_dirs)
        actual_cost = read_cost()
        X[i + 1] = route2x(M, best_route, best_dirs)
        Y[i + 1] = actual_cost

        if (i + 1) % BATCH == 0:
            x = X[: i + 1]
            y = Y[: i + 1]
            regressor.fit(x, y)
            edges = smoothing(M, regressor.coef_)
            
            # if (i + 1) % BATCH*2 == 0:
            #     # M=1 を考慮する
            #     param_Y, param_H = estimate_yh(M, edges)
            #     edges = edges_from_yh(M, param_Y, param_H)
            #     edges = smoothing(M, edges)
            
            G = init_graph(M, edges)

    # export_input(M, "./in/0000.txt", "st13_input.png")
    # export_edges(M, edges, "st13_edges.png")


def parse_arguments():
    parser = argparse.ArgumentParser(description="AHC003 solver.")
    parser.add_argument(
        "--batch",
        type=int,
        default=50,
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
