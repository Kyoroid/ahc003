"""
step1: 辺の重み4250のグラフGを用意する。
step2: グラフGのdijkstra を解き、通過した辺の重みを値 (old + new) / 2 で更新する。
step3: たまに経路を変更する。

score:
89256864392

https://atcoder.jp/contests/ahc003/submissions/22899780
"""


import sys
input = sys.stdin.readline
import heapq
import logging
import math
import random
from pprint import pformat


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


def update_graph(graph, route, dirs, update_func=None):
    for j in range(len(dirs)):
        s = route[j]
        t = route[j+1]
        d = dirs[j]
        graph[s][d] = update_func(graph[s][d])
        graph[t][3-d] = update_func(graph[t][3-d])


def get_route_costs(graph, route, dirs):
    costs = 0
    for j in range(len(dirs)):
        s = route[j]
        d = dirs[j]
        costs += graph[s][d]
    return costs


def inverse_route(M, route, dirs):
    sp, sq = divmod(route[0], M)
    tp, tq = divmod(route[-1], M)
    l1dist = abs(tp - sp) + abs(tq - sq)
    if l1dist < len(dirs):
        return
    DIRECTIONS = [-M, -1, 1, M]
    n = len(dirs)
    for i in range(n // 2):
        dirs[i], dirs[n - i - 1] = dirs[n - i - 1], dirs[i]
    for i in range(n):
        route[i+1] = route[i] + DIRECTIONS[3 - dirs[i]]


def exp1(i, n, factor=10):
    return pow(math.e, -factor * i)


def solve(N = 1000, M = 30):
    G = init_graph(M, 4250)

    for i in range(N):
        s, t = read_query(M)

        prev = dijkstra(M, G, s)
        route, dirs = trace_route(M, prev, s, t)
        if exp1(i, N, 2) > random.random():
            inverse_route(M, route, dirs)

        predicted_cost_g = get_route_costs(G, route, dirs)

        path_length = len(dirs)
        
        write_path(dirs)
        actual_cost_g = read_cost()

        new_g = actual_cost_g // path_length
        update_graph(G, route, dirs, lambda x: (x + new_g) // 2)

    logger.debug(pformat(G))


if __name__ == '__main__':
    random.seed(2021)
    solve()
