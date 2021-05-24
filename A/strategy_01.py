"""
dijkstraの結果をそのまま出力する。ランダム性は持たせない。

score:
63195645751

https://atcoder.jp/contests/ahc003/submissions/22786819
"""

import sys
input = sys.stdin.readline
import heapq
import logging


logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


def read_query(M):
    sy, sx, ty, tx = map(int, input().strip().split())
    return sy * M + sx, ty * M + tx


def read_cost():
    return int(input().strip())


def dijkstra(M, G, start, INF = 10**7):
    MM = M * M
    DIRECTIONS = [-M, -1, 1, M]
    cost = [INF for i in range(MM)]
    q = []
    cost[start] = 0
    heapq.heappush(q, (cost[start], start))
    while q:
        _, u = heapq.heappop(q)
        for i in range(4):
            v = u + DIRECTIONS[i]
            if v < 0 or MM <= v or (u % M == 0) and (v % M == M-1) or (v % M == 0) and (u % M == M-1):
                continue
            if cost[v] > cost[u] + G[u][i]:
                cost[v] = cost[u] + G[u][i]
                heapq.heappush(q, (cost[v], v))
    return cost


def trace_route(M, G, cost, start, end):
    MM = M * M
    DIRECTIONS = [-M, -1, 1, M]
    DIRECTIONS_STR = "DRLU"
    u = end
    ds = "@"
    route = [u]
    dirs = []
    while u != start:
        for i in range(4):
            v = u + DIRECTIONS[i]
            if v < 0 or MM <= v or (u % M == 0) and (v % M == M-1) or (v % M == 0) and (u % M == M-1):
                continue
            if cost[u] == cost[v] + G[u][i]:
                u = v
                ds = DIRECTIONS_STR[i]
                break
        route.append(u)
        dirs.append(ds)
    return route[::-1], dirs[::-1]


def init_graph(M):
    MM = M * M
    v = 1
    G = [[v, v, v, v] for i in range(MM)]
    return G


def solve(N = 1000, M = 30):
    G = init_graph(M)
    for i in range(N):
        s, t = read_query(M)
        cost = dijkstra(M, G, s)
        route, dirs = trace_route(M, G, cost, s, t)
        d_line = "".join(dirs)

        logger.info((divmod(s, M), divmod(t, M), d_line, len(d_line)))
        print("".join(dirs), flush=True)
        _ = read_cost()


if __name__ == '__main__':
    # random.seed(2021)
    solve()
