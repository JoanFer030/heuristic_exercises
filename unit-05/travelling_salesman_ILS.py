import time
import gzip
import math
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

def tour_length(tour, distances):
    """
    Calculate total length of TSP tour
    """
    n = len(tour)
    length = 0
    for i in range(n):
        length += distances[tour[i], tour[(i + 1) % n]]
    return length

def construct_initial_solution(n):
    """
    Generate initial random solution (permutation)
    """
    permutation = list(range(n))
    random.shuffle(permutation)
    return permutation

def stochastic_two_opt(perm):
    """
    Apply stochastic 2-opt operator to a permutation
    """
    result = perm.copy()  # Make a copy to avoid modifying original
    size = len(result)
    # Select two random points in the tour
    p1, p2 = random.randrange(0, size), random.randrange(0, size)
    # Ensure p2 is not adjacent to p1
    exclude = set([p1])
    if p1 == 0:
        exclude.add(size - 1)
    else:
        exclude.add(p1 - 1)
    
    if p1 == size - 1:
        exclude.add(0)
    else:
        exclude.add(p1 + 1)
    
    while p2 in exclude:
        p2 = random.randrange(0, size)
    if p2 < p1:
        p1, p2 = p2, p1
    result[p1:p2] = reversed(result[p1:p2])
    return result

def double_bridge_move(perm):
    """
    Double-bridge move perturbation (4-opt move)
    Creates a stronger perturbation than 2-opt
    """
    size = len(perm)
    slice_length = size // 4
    
    p1 = 1 + random.randrange(0, slice_length)
    p2 = p1 + 1 + random.randrange(0, slice_length)
    p3 = p2 + 1 + random.randrange(0, slice_length)
    
    # Combine slices in new order: a + d + c + b
    return perm[0:p1] + perm[p3:] + perm[p2:p3] + perm[p1:p2]

def local_search(solution, cost, distances, max_no_improve=50):
    """
    Local search procedure using stochastic 2-opt
    """
    count = 0
    current_solution = solution.copy()
    current_cost = cost
    
    while count < max_no_improve:
        new_solution = stochastic_two_opt(current_solution)
        new_cost = tour_length(new_solution, distances)
        if new_cost < current_cost:
            current_solution = new_solution
            current_cost = new_cost
            count = 0  # Reset counter when improvement is found
        else:
            count += 1
    
    return current_solution, current_cost

def perturbation(solution, distances):
    """
    Perturbation phase using double-bridge move
    """
    new_solution = double_bridge_move(solution)
    new_cost = tour_length(new_solution, distances)
    return new_solution, new_cost

def ils_tsp(distances, max_iterations=10000, max_no_improve=50):
    """
    Iterated Local Search algorithm for TSP
    """
    history = []
    n = len(distances)
    
    # 1. Generate initial solution
    best_solution = construct_initial_solution(n)
    best_cost = tour_length(best_solution, distances)
    # 2. Refine with local search
    best_solution, best_cost = local_search(best_solution, best_cost, distances, max_no_improve)
    print(f"Initial solution cost: {best_cost:.2f}")
    history.append((0, best_cost))
    
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        new_solution, new_cost = perturbation(best_solution, distances)
        new_solution, new_cost = local_search(new_solution, new_cost, distances, max_no_improve)
        if new_cost < best_cost:
            best_solution = new_solution
            best_cost = new_cost
            history.append((iteration, best_cost))
    return best_solution, best_cost, history

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
    
    plt.title(f"ILS Solution for {instance_name} (Cost: {cost:.2f})", fontsize=14, fontweight="bold")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(f"./unit-05/output/{instance_name}_ILS.png")

def plot_convergence(cost_history, instance_name):
    iterations, costs = zip(*cost_history)
    plt.figure(figsize = (10, 6))

    plt.plot(iterations, costs, "b-", linewidth=2, label = f"ILS: {min(costs):.2f}")
    plt.xlabel("Iteration")
    plt.ylabel("Best Tour Length")
    plt.title(f"Travelling Salesman Problem - {instance_name} - ILS Convergence")
    plt.xscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"./unit-05/output/{instance_name}_convergence_ILS.png")

# Main execution
if __name__ == "__main__":
    max_iterations = 10000
    max_no_improve = 50
    instance_name = "u159"

    coords = parse_tsp(instance_name)
    dist_matrix = compute_distance_matrix(coords)
    
    t0 = time.time()
    best_tour, best_cost, cost_history = ils_tsp(dist_matrix, max_iterations, max_no_improve)
    elapsed_time = time.time() - t0

    print(f"Instance: {instance_name}")
    print(f"Elapsed time: {elapsed_time:.4f}s")
    print(f"Best tour length: {best_cost:.2f}")
    print(f"Number of iterations: {cost_history[-1][0]}")
    plot_tour(coords, best_tour, best_cost, instance_name)
    plot_convergence(cost_history, instance_name)