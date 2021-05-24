import sys
input = sys.stdin.readline
import heapq
import logging
import random
from pprint import pformat
import matplotlib.pyplot as plt
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_query(M):
    sy, sx, ty, tx = map(int, input().strip().split())
    return sy * M + sx, ty * M + tx


def read_cost():
    return int(input().strip())


def dijkstra(M, G, start, INF = 10**9):
    MM = M * M
    DIRECTIONS = [-M, -1, 1, M]
    cost = [INF for i in range(MM)]
    prev = [-1 for i in range(MM)]
    q = []
    cost[start] = 0
    heapq.heappush(q, (cost[start], start))
    while q:
        u_cost, u = heapq.heappop(q)
        if cost[u] < u_cost:
            continue
        for i in range(4):
            v = u + DIRECTIONS[i]
            if v < 0 or MM <= v or (u % M == 0) and (v % M == M-1) or (v % M == 0) and (u % M == M-1):
                continue
            if cost[v] > cost[u] + G[u][i]:
                cost[v] = cost[u] + G[u][i]
                prev[v] = u
                heapq.heappush(q, (cost[v], v))
    return cost, prev


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


def init_graph(M, init_value):
    MM = M * M
    if init_value >= 0:
        G = [[init_value for j in range(4)] for i in range(MM)]
    else:
        G = [[random.randint(3000, 7000) for j in range(4)] for i in range(MM)]
    return G


def write_path(dirs):
    DIRECTIONS_STR = "DRLU"
    print("".join([DIRECTIONS_STR[i] for i in dirs]), flush=True)


def solve(N = 1000, M = 30):
    G = init_graph(M, 4250)

    x = np.arange(0, N)
    y = np.zeros(N, dtype=np.float)

    for i in range(N):
        s, t = read_query(M)

        cost, prev = dijkstra(M, G, s)
        route, dirs = trace_route(M, prev, s, t)
        predicted_cost = cost[t]
        
        write_path(dirs)
        actual_cost = read_cost()
        ab = predicted_cost / actual_cost
        y[i] = ab

        new_score = actual_cost // len(dirs)

        for j in range(len(route) - 1):
            s = route[j]
            t = route[j+1]
            d = dirs[j]
            G[s][d] = (G[s][d] + new_score) // 2
            G[t][3-d] = (G[t][3-d] + new_score) // 2
    
    pf1 = np.polyfit(x, y, 2)
    pfy1 = np.poly1d(pf1)(x)
    plt.plot(x, y, "x")
    plt.plot(x, pfy1)
    plt.xlabel("query")
    plt.ylabel("predicted length / actual length")
    plt.savefig("fig1_dijkstra_opt_ratio.png")


if __name__ == '__main__':
    random.seed(2021)
    solve()
