import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class Agent:
    total_cost: float
    accuracy: float


def cross(point_o: Agent, point_a: Agent, point_b: Agent) -> int:
    return (point_a.total_cost - point_o.total_cost) * (point_b.accuracy - point_o.accuracy) - (point_a.accuracy - point_o.accuracy) * (point_b.total_cost - point_o.total_cost)

def compute_hull_side(points: list[Agent]) -> list[Agent]:
    hull: list[Agent] = []
    for p in points:
        while len(hull) >= 2 and cross(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)
    return hull

def is_pareto_efficient(others, candidate):
    for other in others:
        if (other.total_cost <= candidate.total_cost and other.accuracy >= candidate.accuracy) and \
           (other.total_cost < candidate.total_cost or other.accuracy > candidate.accuracy):
            return False
    return True

def compute_pareto_frontier(points: list[Agent]) -> list[Agent]:
    points = sorted(list(points), key=lambda p: (p.total_cost, p.accuracy))
    if len(points) <= 1:
        return points

    upper_convex_hull = compute_hull_side(list(reversed(points)))
    pareto_frontier = [agent for agent in upper_convex_hull if is_pareto_efficient(upper_convex_hull, agent)]

    return pareto_frontier

