"""
ロールバック、Numba捨てた
"""


import sys

input = sys.stdin.readline
import logging
import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge


logging.basicConfig(level=logging.INFO)
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


def smoothing(M, edges, window_size=7):
    assert window_size % 2 == 1
    M1M = (M - 1) * M
    x = edges
    x = (x - x.min()) / (x.max() - x.min())
    x = x * 8000 + 1000
    window = np.ones(window_size)
    window /= np.trapz(window)
    for i in range(M):
        # smooth horizontal lane
        l = i * (M - 1)
        r = l + M - 1
        horizontal_in = np.pad(
            x[l:r], 
            (window_size-1, window_size-1),
            "edge"
        )
        x[l:r] = np.convolve(horizontal_in, window, mode="same")[window_size-1:-window_size+1]

        # smooth vertical lane
        l = M1M + i
        r = M1M * 2
        vertical_in = np.pad(
            x[l:r:M], 
            (window_size-1, window_size-1),
            "edge"
        )
        x[l:r:M] = np.convolve(vertical_in, window, mode="same")[window_size-1:-window_size+1]
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


def export_plot(M, x, outfile: str, cmin=1000, cmax=9000):
    import matplotlib.pyplot as plt
    M1M = (M-1) * M

    plt.plot(x)
    plt.ylim(0, 500)
    plt.title(str(x.mean()))
    plt.savefig(outfile)
    plt.close()
    logger.info(f"Save plot from x.")


def solve(N=1000, M=30):
    BATCH = 25
    M1M = (M - 1) * M

    regressor = Ridge(solver="lsqr", tol=1e-3)
    X = np.empty((N + 1, M1M * 2), dtype=np.float32)
    Y = np.empty(N + 1, dtype=np.float32)
    # レーン上の観測値の標準偏差
    LANE_STD = np.empty((N // BATCH, M+M-1), dtype=np.float32)
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
        edges = smoothing(M, regressor.coef_, 15)
        G = update_graph_from(M, G, edges)
        
        for i in range(M-1):
            # horizontal lane
            l = i * (M - 1)
            r = l + M - 1
            LANE_STD[k, i] = edges[l:r].std()
        for i in range(M-1, M+M-1):
            # vertical lane
            l = M1M + i
            r = M1M * 2
            LANE_STD[k, i] = edges[l:r:M].std()
    
    warmup_end = (N // 4) // BATCH
    predictor_end = (N // 2) // BATCH
    export_plot(M, LANE_STD[warmup_end:predictor_end].mean(axis=1), "st9_plot.png".format(k))

if __name__ == "__main__":
    import time

    s = time.time()
    solve()
    t = time.time()
    logger.info(t - s)
