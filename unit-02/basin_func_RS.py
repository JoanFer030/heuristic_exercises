import time
import random
import matplotlib.pyplot as plt
import numpy as np

def basin_function(vector):
    """
    Calculate the sum of squares of all elements in the vector
    f(x) = sum(x_i^2)
    """
    total = 0
    for item in vector:
        total += item ** 2
    return total

def random_sol(search_space, problem_size):
    """
    Generate a random solution within the search space
    """
    min_val = search_space[0]
    max_val = search_space[1]
    input_values = []
    for i in range(problem_size):
        input_values.append(min_val + (max_val - min_val) * random.random())
    return input_values

def search_sol_RS(function, search_space, max_iter, problem_size):
    """
    Search for the optimal solution for the funtion using random search.
    """
    best_cost = float("inf")
    best_sol = None
    samples = []  # store all points
    history = []  # store best solutions found over time

    for _ in range(max_iter):
        new_sol = random_sol(search_space, problem_size)
        new_cost = function(new_sol)
        samples.append(new_sol)

        if new_cost < best_cost:
            best_sol = new_sol
            best_cost = new_cost
        history.append(best_sol)

    result = {
        "best_cost": best_cost,
        "best_solution": best_sol,
        "samples": samples,
        "history": history
    }
    return result

#######################################################################
###                         SOLVING & RESULTS                       ###
#######################################################################
"""
f(x) = sum_{i=1}^nx^2_i

-5.0 < x_i < 5.0 and n=2
"""
search_space = [-5, 5]

t0 = time.time()
result = search_sol_RS(basin_function, search_space = search_space, 
                                       max_iter = 10000, 
                                       problem_size = 2)
elapsed_time = time.time() - t0

best_solution = result["best_solution"]
best_cost = result["best_cost"]
history = result["history"]
samples = result["samples"]

print(f"Best solution found: {best_solution} with cost: {best_cost}")
print(f"Number of iterations: {len(history)}")
print(f"Elapsed time: {elapsed_time}s")

# Plot the convergence
costs_history = [basin_function(sol) for sol in history]
plt.figure(figsize=(10, 6))
plt.step(costs_history, "b-", linewidth=2)
plt.xlabel("Iterations")
plt.xscale("log")
plt.ylabel("Best Cost Found")
plt.title("Random Search for $f(x) = \sum_{i=1}^nx^2_i$")
plt.grid(True)
plt.savefig("./unit-02/output/basin_convergence.png")


# Plot the search space and solutions (for 2D case)
x = np.linspace(search_space[0], search_space[1], 100)
y = np.linspace(search_space[0], search_space[1], 100)
X, Y = np.meshgrid(x, y)
Z = np.array([[basin_function([xx, yy]) for xx in x] for yy in y])
samples_array = np.array(samples)
best_array = np.array(history)

plt.figure(figsize=(10, 8))
plt.contourf(X, Y, Z, levels=50, cmap="viridis", alpha=0.6)
plt.colorbar(label="f(x)")
plt.scatter(samples_array[:, 0], samples_array[:, 1], c="blue", alpha=0.1, s=1, label="All samples") # Plot all points
plt.scatter(best_array[:, 0], best_array[:, 1], c="red", s=30, alpha=0.7, label="Improving solutions")
plt.scatter(best_array[-1, 0], best_array[-1, 1], c="yellow", s=100, marker="*", edgecolors="black", label="Final best solution")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("\nRandom Search Convergence for $f(x) = \sum_{i=1}^nx^2_i$\n$\\forall -5.0 < x_i < 5.0\\text{ and } n=2$")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("./unit-02/output/basin_search.png")