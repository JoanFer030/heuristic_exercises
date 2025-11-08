import random
import time
import matplotlib.pyplot as plt
import numpy as np


def generate_knapsack_problem(n_items, capacity_ratio, seed=None):
    """
    Generate a random knapsack problem instance
    """
    if seed is not None:
        random.seed(seed)
    
    weights = [random.randint(1, 100) for _ in range(n_items)]
    values = [random.randint(1, 100) for _ in range(n_items)]
    
    total_weight = sum(weights)
    capacity = int(total_weight * capacity_ratio)
    
    return weights, values, capacity


def random_knapsack(weights, values, capacity, iterations=100000):
    """
    Random Search algorithm for the 0-1 knapsack problem
    """
    n = len(values)
    best_solution = [0] * n
    best_value = 0
    best_weight = 0
    value_history = []  # Track best values over iterations
    
    for i in range(iterations):
        candidate = [random.randint(0, 1) for _ in range(n)]
        total_weight = sum(candidate[i] * weights[i] for i in range(n))
        total_value = sum(candidate[i] * values[i] for i in range(n))
        
        if total_weight <= capacity and total_value > best_value:
            best_value = total_value
            best_solution = candidate.copy()
            best_weight = total_weight
        
        value_history.append(best_value)
    
    return best_solution, best_value, best_weight, value_history


def greedy_knapsack(weights, values, capacity):
    """
    Greedy construction algorithm for the 0-1 knapsack problem
    using value-to-weight ratio
    """
    n = len(values)
    # Compute value/weight ratio
    items = [(i, values[i] / weights[i], values[i], weights[i]) for i in range(n)]
    # Sort items by ratio in descending order
    items.sort(key=lambda x: x[1], reverse=True)
    
    total_value = 0
    total_weight = 0
    solution = [0] * n
    
    for i, _, value, weight in items:
        if total_weight + weight <= capacity:
            solution[i] = 1
            total_weight += weight
            total_value += value
    
    return solution, total_value, total_weight


def plot_knapsack_comparison(rs_values, greedy_value, title):
    """
    Plot the convergence of random search compared to greedy solutions
    """
    iterations = list(range(0, len(rs_values)))
    
    plt.figure(figsize = (10, 6))
    plt.plot(iterations, rs_values, "b-", linewidth = 2, label = f"Random Search: {max(rs_values)}")
    plt.axhline(y=greedy_value, color="r", linestyle="--", linewidth=2, label=f"Greedy Algorithm: {greedy_value}")
    plt.xlabel("Iterations")
    plt.ylabel("Best Value Found")
    plt.title(title)
    plt.legend()
    plt.xscale("log")
    plt.grid(True, alpha=0.3)
    plt.savefig("./unit-02/output/knapsack_convergence_RS.png")


#######################################################################
###                         SOLVING & RESULTS                       ###
#######################################################################
size = 20
max_iterations = 100000

# Generate larger problem instances
weights, values, capacity = generate_knapsack_problem(size, capacity_ratio=0.2)


# Run Random Search
t0 = time.time()
rs_solution, rs_value, rs_weight, rs_history = random_knapsack(weights, values, capacity, max_iterations)
rs_time = time.time() - t0

# Run Greedy Algorithm
t0 = time.time()
greedy_solution, greedy_value, greedy_weight = greedy_knapsack(weights, values, capacity)
greedy_time = time.time() - t0


# Print summary results for larger instances
print(f"{"Algorithm":<15} {"Value":<10} {"Weight":<10} {"Time (s)":<10}")
print("-" * 45)
print(f"{"Random Search":<15} {rs_value:<10} {rs_weight:<10.1f} {rs_time:<10.4f}")
print(f"{"Greedy":<15} {greedy_value:<10} {greedy_weight:<10.1f} {greedy_time:<10.4f}")

# Plot convergence for larger instances
plot_knapsack_comparison(rs_history, greedy_value, f"Knapsack Problem - {size} items")