"""Bounded four-neighbour A* with deterministic ties and no corner cutting."""

import heapq


def plan_cells(blocked, start, goal, bounds):
    width, height = bounds
    if width * height > 250000:
        raise ValueError("map exceeds planner budget")

    def valid(p):
        return 0 <= p[0] < width and 0 <= p[1] < height and p not in blocked

    if not valid(start) or not valid(goal):
        return None
    queue = [(0, 0, start)]
    costs = {start: 0}
    previous = {}
    while queue:
        _, cost, current = heapq.heappop(queue)
        if cost != costs[current]:
            continue
        if current == goal:
            route = [current]
            while current in previous:
                current = previous[current]
                route.append(current)
            return list(reversed(route))
        x, y = current
        for nxt in ((x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)):
            new = cost + 1
            if valid(nxt) and new < costs.get(nxt, float("inf")):
                costs[nxt] = new
                previous[nxt] = current
                heapq.heappush(
                    queue,
                    (new + abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1]), new, nxt),
                )
    return None
