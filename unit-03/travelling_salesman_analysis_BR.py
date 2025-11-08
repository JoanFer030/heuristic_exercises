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
    with gzip.open(f".data/tsp/{instance_name}.tsp.gz", "r") as f:
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

def enhanced_geometric_distribution(k: int, alpha: float, method: str = "standard") -> list[float]:
    """
    Enhanced geometric distribution with different generation methods
    """
    if method == "standard": # Standard geometric distribution
        probs = [(1 - alpha) * (alpha ** i) for i in range(k)]
    elif method == "exponential": # Exponential decay with smoother curve
        probs = [math.exp(-alpha * i) for i in range(k)]
    elif method == "adaptive":
        base_alpha = min(alpha, 0.8)  # Cap alpha for stability
        probs = [(1 - base_alpha) * (base_alpha ** (i * (2/k))) for i in range(k)]
    # Normalize probabilities
    total = sum(probs)
    if total > 0:
        return [p / total for p in probs]
    else:
        return [1.0 / k] * k

def biased_randomized_tsp(distances, start = 0, alpha = 0.2, method = "standard"):
    """
    Enhanced Biased-Randomized Algorithm for TSP
    """
    n = len(distances)
    tour = [start]
    unvisited = set(range(n)) - {start}
    while unvisited:
        last = tour[-1]
        sorted_candidates = sorted(unvisited, key = lambda city: distances[last, city])
        k = len(sorted_candidates)
        # Use enhanced geometric distribution
        probs = enhanced_geometric_distribution(k, alpha, method)
        # Select next city using biased probabilities
        next_city = random.choices(sorted_candidates, weights=probs, k=1)[0]
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

def analyze_start_cities(distances, num_runs=100, alpha=0.1):
    """
    Analyze performance when starting from different cities
    """
    n = len(distances)
    results = []
    for start_city in range(n): 
        costs = []
        for _ in range(num_runs):
            tour = biased_randomized_tsp(distances, start = start_city, alpha = alpha)
            cost = tour_length(tour, distances)
            costs.append(cost)
        results.append((start_city, np.mean(costs), np.std(costs)))
    return results

def analyze_alpha_range(distances, alpha_values, num_runs_per_alpha=50):
    """
    Study the best range value for alpha parameter
    """
    results = []
    for alpha in alpha_values:
        best_cost = float("inf")
        alpha_costs = []
        for _ in range(num_runs_per_alpha):
            start_city = random.randint(0, len(distances) - 1)
            tour = biased_randomized_tsp(distances, start = start_city, alpha = alpha)
            cost = tour_length(tour, distances)
            alpha_costs.append(cost)
            if cost < best_cost:
                best_cost = cost
        avg_cost = np.mean(alpha_costs)
        std_cost = np.std(alpha_costs)
        results.append((alpha, best_cost, avg_cost, std_cost))
    return results

def compare_geometric_methods(distances, num_runs=100, alpha=0.1):
    """
    Compare different geometric generation methods
    """
    methods = ["standard", "exponential", "adaptive"]
    results = {}
    for method in methods:
        method_costs = []
        for _ in range(num_runs):
            start_city = random.randint(0, len(distances) - 1)
            tour = biased_randomized_tsp(distances, start = start_city, alpha = alpha, method = method)
            cost = tour_length(tour, distances)
            method_costs.append(cost)
        best_cost = min(method_costs)
        avg_cost = np.mean(method_costs)
        std_cost = np.std(method_costs)
        results[method] = {
            "best": best_cost,
            "average": avg_cost,
            "std": std_cost,
            "all_costs": method_costs
        }
    return results

def plot_comprehensive_analysis(start_city_results, alpha_results, method_results, instance_name):
    """
    Create comprehensive visualization of all analyses
    """
    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Different starting cities
    cities, avg_costs, std_costs = zip(*start_city_results)
    ax1.bar(cities, avg_costs, yerr = std_costs, alpha = 0.7, color = "skyblue", capsize = 2, ecolor = "gray")
    ax1.axhline(y=min(avg_costs), color = "red", linestyle = "--", label = f"Best: {min(avg_costs):.2f}")
    ax1.set_xlabel("Starting City")
    ax1.set_ylabel("Tour Cost")
    ax1.set_title("Performance by Starting City")
    ax1.legend()
    ax1.grid(True, alpha = 0.3)
    
    # Plot 2: Alpha parameter analysis
    alphas, best_costs, avg_costs, std_costs = zip(*alpha_results)
    ax2.errorbar(alphas, avg_costs, yerr = std_costs, fmt = "-o", capsize = 5, label = "Average ± Std Dev", alpha = 0.7)
    ax2.plot(alphas, best_costs, "ro-", label = "Best Cost")
    ax2.set_xlabel("Alpha Value")
    ax2.set_ylabel("Tour Cost")
    ax2.set_title("Alpha Parameter Analysis")
    ax2.legend()
    ax2.grid(True, alpha = 0.3)
    
    # Plot 3: Method comparison
    methods = list(method_results.keys())
    best_values = [method_results[m]["best"] for m in methods]
    avg_values = [method_results[m]["average"] for m in methods]
    
    x_pos = np.arange(len(methods))
    width = 0.35
    
    ax3.bar(x_pos - width/2, best_values, width, label = "Best Cost", alpha = 0.7)
    ax3.bar(x_pos + width/2, avg_values, width, label = "Average Cost", alpha = 0.7)
    ax3.set_xlabel("Geometric Method")
    ax3.set_ylabel("Tour Cost")
    ax3.set_title("Geometric Generation Methods Comparison")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(methods, rotation = 45)
    ax3.legend()
    ax3.grid(True, alpha = 0.3)
    
    plt.tight_layout()
    plt.savefig(f"./unit-03/output/{instance_name}_analysis_BR.png")

def print_summary_statistics(alpha_results, method_results):
    """
    Print comprehensive summary statistics
    """
    print("\n" + "="*60)
    print("COMPREHENSIVE ANALYSIS SUMMARY")
    print("="*60)
    
    # Best alpha results
    best_alpha_result = min(alpha_results, key=lambda x: x[1])
    print(f"\nBest Alpha Parameter:")
    print(f"  Value: {best_alpha_result[0]:.2f}")
    print(f"  Best Cost: {best_alpha_result[1]:.2f}")
    print(f"  Average Cost: {best_alpha_result[2]:.2f}")
    
    # Best method
    best_method = min(method_results.items(), key=lambda x: x[1]["best"])
    print(f"\nBest Geometric Method:")
    print(f"  Method: {best_method[0]}")
    print(f"  Best Cost: {best_method[1]["best"]:.2f}")
    print(f"  Average Cost: {best_method[1]["average"]:.2f}")


#######################################################################
###                         MAIN ANALYSIS                          ###
#######################################################################
if __name__ == "__main__":
    instance_name = "berlin52"
    
    coords = parse_tsp(instance_name)
    dist_matrix = compute_distance_matrix(coords)
    
    # Different starting cities
    start_city_results = analyze_start_cities(dist_matrix, num_runs=100, alpha=0.1)
    # Alpha parameter range
    alpha_values = np.arange(0.025, 0.51, 0.05) 
    alpha_results = analyze_alpha_range(dist_matrix, alpha_values, num_runs_per_alpha=30)
    
    # Geometric generation methods (using best alpha)
    best_alpha = min(alpha_results, key=lambda x: x[1])[0]
    method_results = compare_geometric_methods(dist_matrix, num_runs=50, alpha=best_alpha)
    
    # Results
    plot_comprehensive_analysis(start_city_results, alpha_results, method_results, instance_name)
    print_summary_statistics(alpha_results, method_results)
