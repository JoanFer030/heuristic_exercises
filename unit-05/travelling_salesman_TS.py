import time
import gzip
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

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

def stochastic_two_opt_with_edges(perm):
    """
    Apply stochastic 2-opt operator and return the solution with edges that were modified
    """
    result = perm.copy()
    size = len(result)
    
    p1, p2 = random.randrange(0, size), random.randrange(0, size)
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
    edge1 = [perm[p1 - 1], perm[p1]] if p1 > 0 else [perm[size - 1], perm[0]]
    edge2 = [perm[p2 - 1], perm[p2]] if p2 > 0 else [perm[size - 1], perm[0]]
    result[p1:p2] = reversed(result[p1:p2])
    
    return result, [edge1, edge2]

def generate_neighbors(base_solution, distances, tabu_list, num_neighbors=20):
    """
    Generate neighbor solutions avoiding tabu moves
    """
    neighbors = []
    
    for _ in range(num_neighbors):
        new_solution, edges_removed = stochastic_two_opt_with_edges(base_solution)
        new_cost = tour_length(new_solution, distances)
        neighbor = {
            "permutation": new_solution,
            "cost": new_cost,
            "edges_removed": edges_removed
        }
        neighbors.append(neighbor)
    
    return neighbors

def is_tabu(solution, tabu_list):
    """
    Check if a solution contains any tabu edges
    """
    size = len(solution)
    for i in range(size):
        if i == size - 1:
            edge = [solution[i], solution[0]]
        else:
            edge = [solution[i], solution[i + 1]]
        if edge in tabu_list or [edge[1], edge[0]] in tabu_list:
            return True
    
    return False

def locate_best_neighbor(neighbors, tabu_list, best_solution_cost):
    """
    Find the best non-tabu neighbor, applying aspiration criterion
    """
    # Sort neighbors by cost
    sorted_neighbors = sorted(neighbors, key=lambda x: x["cost"])
    best_candidate = None    
    for neighbor in sorted_neighbors:
        if neighbor["cost"] < best_solution_cost:
            best_candidate = neighbor
            break
        if not is_tabu(neighbor["permutation"], tabu_list):
            best_candidate = neighbor
            break
    if best_candidate is None and sorted_neighbors:
        best_candidate = sorted_neighbors[0]
    
    return best_candidate

def tabu_search_tsp(distances, max_iterations=1000, tabu_tenure=15, num_neighbors=20):
    """
    Tabu Search algorithm for TSP
    """
    history = []
    n = len(distances)
    
    current_solution = construct_initial_solution(n)
    current_cost = tour_length(current_solution, distances)
    
    best_solution = current_solution.copy()
    best_cost = current_cost
    print(f"Initial solution cost: {best_cost:.2f}")
    tabu_list = deque(maxlen=tabu_tenure)

    history.append((0, best_cost))
    for iteration in range(1, max_iterations + 1):
        neighbors = generate_neighbors(current_solution, distances, tabu_list, num_neighbors)
        best_candidate = locate_best_neighbor(neighbors, tabu_list, best_cost)
        
        if best_candidate is None:
            continue
        current_solution = best_candidate["permutation"]
        current_cost = best_candidate["cost"]
        if current_cost < best_cost:
            best_solution = current_solution.copy()
            best_cost = current_cost
            history.append((iteration, best_cost))
        for edge in best_candidate["edges_removed"]:
            tabu_list.append(edge)
            tabu_list.append([edge[1], edge[0]])
    
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

    for i, (x, y) in enumerate(coords):
        plt.text(x, y, str(i + 1), fontsize = 8, ha = "center", va = "center", 
                 bbox = {"boxstyle": "circle", "pad":0.2, "facecolor": "white", "alpha": 0.7})
    plt.plot(start_x, start_y, "gs", markersize = 10, markerfacecolor = "none", markeredgewidth = 2, label = f"Start City (Node {tour[0]+1})")
    
    plt.title(f"Tabu Search Solution for {instance_name} (Cost: {cost:.2f})", fontsize=14, fontweight="bold")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(f"./unit-05/output/{instance_name}_TS.png")

def plot_convergence(cost_history, instance_name):
    iterations, costs = zip(*cost_history)
    plt.figure(figsize = (10, 6))

    plt.plot(iterations, costs, "r-", linewidth=2, label = f"Tabu Search: {min(costs):.2f}")
    plt.xlabel("Iteration")
    plt.ylabel("Best Tour Length")
    plt.title(f"Travelling Salesman Problem - {instance_name} - Tabu Search Convergence")
    plt.xscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"./unit-05/output/{instance_name}_convergence_TS.png")

if __name__ == "__main__":
    instance_name = "u159"
    # Tabu Search parameters
    max_iterations = 10000
    tabu_tenure = 15  # Size of tabu list
    num_neighbors = 20  # Number of neighbors to generate at each iteration
    
    coords = parse_tsp(instance_name)
    dist_matrix = compute_distance_matrix(coords)
    t0 = time.time()
    best_tour, best_cost, cost_history = tabu_search_tsp(dist_matrix, max_iterations, tabu_tenure, num_neighbors)
    elapsed_time = time.time() - t0
    
    print(f"Instance: {instance_name}")
    print(f"Tabu tenure (Size of tabu list): {tabu_tenure}")
    print(f"Number of neighbors to generate at each iteration: {num_neighbors}")
    print(f"Elapsed time: {elapsed_time:.4f}s")
    print(f"Best tour length: {best_cost:.2f}")
    print(f"Number of iterations: {cost_history[-1][0]}")
    plot_tour(coords, best_tour, best_cost, instance_name)
    plot_convergence(cost_history, instance_name)