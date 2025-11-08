

import gzip
import math
import time
import random
import numpy as np
import matplotlib.pyplot as plt


def parse_tsp(instance_name: str):
    """
    Parse TSP instance file in TSPLIB format
    """
    coords = []
    with gzip.open(f"./data/tsp/{instance_name}.tsp.gz", "r") as f:
        lines = [l.decode().strip() for l in f.readlines()]
        start = lines.index("NODE_COORD_SECTION")
        end = lines.index("EOF")
        for line in lines[start+1:end]:
            parts = line.split(" ")
            x, y = float(parts[1]), float(parts[2])
            coords.append((x, y))
    return np.array(coords)

def compute_distance_matrix(coords):
    """
    Compute Euclidean distance matrix between all points
    """
    n = len(coords)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i, j] = math.hypot(coords[i][0] - coords[j][0],
                                             coords[i][1] - coords[j][1])
    return dist_matrix

def greedy_tsp(distances, start=0):
    """
    Deterministic greedy algorithm for TSP
    Always chooses the nearest unvisited city
    """
    n = len(distances)
    tour = [start]
    unvisited = set(range(n)) - {start}
    while unvisited:
        last = tour[-1]
        # Find the closest unvisited city
        next_city = min(unvisited, key=lambda city: distances[last, city])
        tour.append(next_city)
        unvisited.remove(next_city)
    return tour

def biased_random_choice(candidates, probs):
    """
    Select one element from candidates using given probabilities
    """
    return random.choices(candidates, weights=probs, k=1)[0]

def biased_randomized_tsp(distances, start=0, alpha=0.2):
    """
    Biased-Randomized Algorithm for TSP
    Uses geometric distribution to bias selection toward nearest cities
    """
    n = len(distances)
    tour = [start]
    unvisited = set(range(n)) - {start}
    while unvisited:
        last = tour[-1]
        sorted_candidates = sorted(unvisited, key=lambda city: distances[last, city])
        k = len(sorted_candidates)
        probs = [(1 - alpha) * (alpha ** i) for i in range(k)]
        total = sum(probs)
        probs = [p / total for p in probs]
        next_city = biased_random_choice(sorted_candidates, probs)
        tour.append(next_city)
        unvisited.remove(next_city)
    
    return tour

def tour_length(tour, distances):
    """
    Calculate total length of TSP tour
    """
    n = len(tour)
    length = 0
    for i in range(n):
        length += distances[tour[i], tour[(i + 1) % n]]
    return length

def plot_tour(coords, tour, cost, instance_name):
    """
    Visualize TSP tour
    """
    start_x, start_y = coords[tour[0]]
    tour_coords = coords[tour + [tour[0]]]

    # Plot
    plt.figure(figsize=(10, 8))
    plt.plot(tour_coords[:, 0], tour_coords[:, 1], "b-", linewidth=2, alpha=0.7)
    plt.plot(tour_coords[:, 0], tour_coords[:, 1], "ro", markersize=6)

    # Labels
    for i, (x, y) in enumerate(coords):
        plt.text(x, y, str(i + 1), fontsize = 8, ha = "center", va = "center", 
                 bbox = {"boxstyle": "circle", "pad":0.2, "facecolor": "white", "alpha": 0.7})
    plt.plot(start_x, start_y, "gs", markersize = 10, markerfacecolor = "none", markeredgewidth = 2, label = f"Start City (Node {tour[0]+1})")
    
    plt.title(f"BRA Solution for {instance_name} (Cost: {cost:.2f})", fontsize=14, fontweight="bold")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(f"./unit-03/output/{instance_name}_BR.png")

def plot_convergence(cost_history, greedy_cost, instance_name):
    iterations, costs = zip(*cost_history)
    plt.figure(figsize = (10, 6))

    plt.plot(iterations, costs, "b-", linewidth=2, label = f"Biased-Randomized: {min(costs):.2f}")
    plt.axhline(y = greedy_cost, color = "r", linestyle = "--", linewidth = 2, label = f"Greedy: {greedy_cost:.2f}")
    plt.xlabel("Iteration")
    plt.ylabel("Best Tour Length")
    plt.title(f"Travelling Salesman Problem - {instance_name}")
    plt.xscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"./unit-03/output/{instance_name}_convergence_BR.png")


def multi_start_bra_tsp(distances, num_runs=1000, alpha=0.1):
    """
    Perform multiple runs of BRA-TSP and return best solution
    """
    best_tour = None
    best_cost = float("inf")
    cost_history = []
    
    for run in range(num_runs):
        # Try different starting cities for diversification
        start_city = random.randint(0, len(distances) - 1)
        tour = biased_randomized_tsp(distances, start=start_city, alpha=alpha)
        cost = tour_length(tour, distances)
        
        if cost < best_cost:
            best_cost = cost
            best_tour = tour
            cost_history.append((run + 1, best_cost))

    return best_tour, best_cost, cost_history


#######################################################################
###                         SOLVING & RESULTS                       ###
#######################################################################
if __name__ == "__main__":
    instance_name = "pr76" # berlin52, ulisses16, u159, pr76, eil101

    print(instance_name)
    coords = parse_tsp(instance_name)
    dist_matrix = compute_distance_matrix(coords)

    # Greedy Solution
    t0 = time.time()
    greedy_tour = greedy_tsp(dist_matrix)
    greedy_cost = tour_length(greedy_tour, dist_matrix)
    print(f"\nGreedy tour length: {greedy_cost:.2f} - Time: {time.time() - t0}s")

    # BRA Solution
    num_runs = 1000
    alpha = 0.10

    t0 = time.time()
    best_tour, best_cost, cost_history = multi_start_bra_tsp(
        dist_matrix, num_runs = num_runs, alpha = alpha
    )
    print(f"Biased-randomized tour length: {best_cost:.2f} ({'+' if best_cost > greedy_cost else '-'}{((greedy_cost - best_cost) / greedy_cost * 100):.2f}%) - Time: {time.time() - t0}s\n")

    plot_convergence(cost_history, greedy_cost, instance_name)
    plot_tour(coords, best_tour, best_cost, instance_name)
