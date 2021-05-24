"""
線形回帰の導入準備
経路のエンコーダを作成
"""


import sys
import math
input = sys.stdin.readline
import heapq
import logging
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_query(M):
    si, sj, ti, tj = map(int, input().strip().split())
    return si * M + sj, ti * M + tj


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
    return prev


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
    G = [[init_value for j in range(4)] for i in range(MM)]
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


def get_route_costs(graph, route, dirs):
    costs = 0
    for j in range(len(dirs)):
        s = route[j]
        d = dirs[j]
        costs += graph[s][d]
    return costs


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


def solve(N = 1000, M = 30):
    G = init_graph(M, 5000)
    
    for i in range(N):
        s, t = read_query(M)
        
        prev = dijkstra(M, G, s)
        best_route, best_dirs = trace_route(M, prev, s, t)

        write_path(best_dirs)
        actual_cost = read_cost()

        update_graph(M, G, best_route, best_dirs, lambda x: (x + actual_cost / len(best_dirs)) // 2, w=1)


if __name__ == '__main__':
    solve()
