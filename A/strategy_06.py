"""
Ridge回帰導入
step1: 辺の重み5000のグラフGを用意する。
step2: 以下の操作を5回繰り返す
step2-1: グラフGの頂点に関する最短経路問題を200回解く
step2-2: グラフGの辺に関する線形回帰モデルを作る
step2-3: モデルの重みを平滑化し、グラフGの新たな辺の重みとする

predicted score:
95.0% confidence interval: 904148437 [900442856, 907854018]

atcual score:
90132495592

https://atcoder.jp/contests/ahc003/submissions/22957265
"""


import sys
import math
input = sys.stdin.readline
import logging
import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
from scipy.signal import convolve
from sklearn.linear_model import Ridge


logging.basicConfig(level=logging.ERROR)
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
    data = [5000] * (M * (M-1) * 2)
    s = []
    t = []
    # horizontal lane
    for i in range(M):
        for j in range(M-1):
            s.append(i*M+j)
            t.append(i*M+j+1)
    # vertical lane
    for i in range(M-1):
        for j in range(M):
            s.append(i*M+j)
            t.append((i+1)*M+j)
    G = csr_matrix((data, (s, t)), shape=(MM, MM))
    return G


def smoothing(M, edges, window_size=7):
    assert window_size % 2 == 1
    x = edges
    MM = M * M
    x = (x - x.min()) / (x.max() - x.min())
    x = x * 8000 + 1000
    window = np.hanning(window_size)
    window_area = np.trapz(window)
    for i in range(M):
        # smooth horizontal lane
        horizontal_in = np.pad(
            x[i*M:(i+1)*M:1], (window_size//2, window_size//2),
            "constant", constant_values=(5000, 5000)
        )
        horizontal_out = convolve(horizontal_in, window, mode="same") / window_area
        x[i*M:(i+1)*M:1] = horizontal_out[window_size//2:M+window_size//2]

        # smooth vertical lane
        vertical_in = np.pad(
            x[MM+i:MM+MM+i:M], (window_size//2, window_size//2),
            "constant", constant_values=(5000, 5000)
        )
        vertical_out = convolve(vertical_in, window, mode="same") / window_area
        x[MM+i:MM+MM+i:M] = vertical_out[window_size//2:M+window_size//2]
    return x


def update_graph_from(M, G, edges):
    # G: csr_matrix (MM, MM)
    # edges: (MM * 2)
    MM = M * M

    data = []
    s = []
    t = []
    # horizontal lane
    for i in range(M):
        for j in range(M-1):
            data.append(edges[i*M+j])
            s.append(i*M+j)
            t.append(i*M+j+1)
    # vertical lane
    for i in range(M-1):
        for j in range(M):
            data.append(edges[MM+i*M+j])
            s.append(i*M+j)
            t.append((i+1)*M+j)
    G = csr_matrix((data, (s, t)), shape=(MM, MM))
    return G


def write_path(dirs):
    DIRECTIONS_STR = "DRLU"
    print("".join([DIRECTIONS_STR[i] for i in dirs]), flush=True)


def update_graph(M, graph, route, dirs, update_func, w=0):
    x = [0 for k in range(len(dirs))]
    c = [0 for k in range(len(dirs))]
    for k in range(len(dirs)):
        si, sj = divmod(route[k], M)
        ti, tj = divmod(route[k+1], M)
        t = route[k+1]
        d = dirs[k]
        if d == 3:
            # TOP (vertical)
            for i in range(ti-w, ti+w+1):
                if 0 <= i < M:
                    x[k] += graph[i*M+tj][0]
                    c[k] += 1
        elif d == 2:
            # LEFT (horizontal)
            for j in range(tj-w, tj+w+1):
                if 0 <= j < M:
                    x[k] += graph[ti*M+j][1]
                    c[k] += 1
        elif d == 1:
            # RIGHT (horizontal)
            for j in range(sj-w, sj+w+1):
                if 0 <= j < M:
                    x[k] += graph[si*M+j][1]
                    c[k] += 1
        elif d == 0:
            # DOWN (vertical)
            for i in range(si-w, si+w+1):
                if 0 <= i < M:
                    x[k] += graph[i*M+sj][0]
                    c[k] += 1
    for k in range(len(dirs)):
        s = route[k]
        t = route[k+1]
        d = dirs[k]
        graph[s][d] = update_func(x[k] // c[k])
        graph[t][3-d] = update_func(x[k] // c[k])


def route2x(M, route, dirs):
    MM = M * M
    x = np.zeros((MM*2), dtype=np.float32)
    for j in range(len(dirs)):
        s = route[j]
        d = dirs[j]
        if d == 3:
            # TOP (vertical)
            x[s+MM-M] = 1.0
        elif d == 2:
            # LEFT (horizontal)
            x[s-1] = 1.0
        elif d == 1:
            # RIGHT (horizontal)
            x[s] = 1.0
        elif d == 0:
            # DOWN (vertical)
            x[s+MM] = 1.0
    return x


def cost2y(cost):
    return np.array([cost, ], dtype=np.float32)


def exp_decay(t, la=0.001):
    return pow(math.e, -la * t)


def export_input(M, infile: str, outfile: str, cmin=1000, cmax=9000):
    import matplotlib.pyplot as plt
    from pathlib import Path
    MM = M * M
    y = np.full((M*2, M*2), np.nan, dtype=np.float32)
    with Path(infile).open(mode="r") as f:
        # horizontal lane
        for i in range(M):
            x = list(map(int, f.readline().split()))
            for j in range(M-1):
                y[i*2, j*2+1] = x[j]
        # vertical lane
        for i in range(M-1):
            x = list(map(int, f.readline().split()))
            for j in range(M):
                y[i*2+1, j*2] = x[j]
    plt.matshow(y, cmap="bwr")
    plt.colorbar()
    plt.clim(cmin, cmax)
    plt.savefig(outfile)
    logger.info(f"Save heatmap from {infile}.")


def export_edges(M, edges, outfile: str, cmin=1000, cmax=9000):
    import matplotlib.pyplot as plt
    MM = M * M
    x = edges
    y = np.full((M*2, M*2), np.nan, dtype=np.float32)
    for i in range(M):
        for j in range(M):
            # horizontal lane
            y[i*2, j*2+1] = x[i*M+j]
            # vertical lane
            y[i*2+1, j*2] = x[MM+i*M+j]
    # plt.imsave(filename, y, cmap="bwr")
    plt.matshow(y, cmap="bwr")
    plt.colorbar()
    plt.clim(cmin, cmax)
    plt.savefig(outfile)
    logger.info(f"Save heatmap from edges.")


def export_graph(M, G, outfile: str, cmin=1000, cmax=9000):
    import matplotlib.pyplot as plt
    MM = M * M
    y = np.full((M*2, M*2), np.nan, dtype=np.float32)
    # horizontal lane
    for i in range(M):
        for j in range(M-1):
            s = i*M+j
            t = i*M+j+1
            y[i*2, j*2+1] = G[s, t]
    # vertical lane
    for i in range(M-1):
        for j in range(M):
            s = i*M+j
            t = (i+1)*M+j
            y[i*2+1, j*2] = G[s, t]
    # plt.imsave(filename, y, cmap="bwr")
    plt.matshow(y, cmap="bwr")
    plt.colorbar()
    plt.clim(cmin, cmax)
    plt.savefig(outfile)
    logger.info(f"Save heatmap from graph.")


def solve(N = 1000, M = 30, INF=100000000):
    BATCH = 200
    regressor = None
    X = np.empty((BATCH+1, M*M*2), dtype=np.float32)
    Y = np.empty(BATCH+1, dtype=np.float32)
    G = init_graph(M)

    for k in range(N // BATCH):
        X[0, :] = 1
        Y[0] = regressor.coef_.sum() if regressor else 5000.0*M*M*2
        
        for i in range(BATCH):
            s, t = read_query(M)
            
            cost, prev = shortest_path(G, directed=False, indices=s, return_predecessors=True)
            best_route, best_dirs = trace_route(M, prev, s, t)

            write_path(best_dirs)
            actual_cost = read_cost()

            X[i+1] = route2x(M, best_route, best_dirs)
            Y[i+1] = cost2y(actual_cost)

        regressor = Ridge()
        regressor.fit(X, Y)
        edges = smoothing(M, regressor.coef_, 11)
        G = update_graph_from(M, G, edges)
    
    # export_input(M, "./in/0000.txt", "st06_hmap_input_0000.png")
    # export_graph(M, G, "st06_hmap_graph_0000.png")
    # export_edges(M, regressor.coef_, "st06_hmap_coef_0000.png")
    # export_edges(M, edges, "st06_hmap_edges_0000.png")


if __name__ == '__main__':
    solve()
