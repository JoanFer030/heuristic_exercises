import random
import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

def load_instance(instance_name: str):
    file_path = f"./data/pop/{instance_name}.txt"
    data = []
    with open(file_path, "r") as file:
        for line in file:
            if not line.startswith("#"):
                data.append(line.strip())
    target_return = np.array(data[0].split(","), dtype = float)
    expected_returns = np.array(data[1].split(","), dtype = float)
    cov_matrix = np.array([row.split(",") for row in data[2:]], dtype=float)

    return target_return, expected_returns, cov_matrix, len(expected_returns)

##########################
## PARAMETERS
##########################
n_generations = 1000
restart_interval = 100
pop_size = 300
crossover_rate = 0.7
mutation_rate = 0.2 
gene_mutation_rate = 0.2
tourn_size = 3
hof_size = 2
penalty = 100

instance_name = "rpop_data_1"
target_return, expected_returns, cov_matrix, n_assets = load_instance(instance_name)

def evaluate_portfolio(individual):
    individual = np.array(individual)
    portfolio_return = np.dot(individual, expected_returns)
    portfolio_risk = np.dot(np.dot(individual, cov_matrix), individual)
    sum_of_weights_cons = 0
    if np.sum(individual) > 1.01 or np.sum(individual) < 0.99:
        sum_of_weights_cons = 1
    return_cons = 0
    if portfolio_return < target_return:
        return_cons = 1
    weight_cons = 0
    if np.any(individual < 0.0) or np.any(individual > 1.0):
        weight_cons = 1
    n_violations = sum_of_weights_cons + return_cons + weight_cons
    portfolio_risk += n_violations * penalty
    return portfolio_risk,

creator.create("FitnessMinimize", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMinimize)

toolbox = base.Toolbox()

toolbox.register("attr_float", random.uniform, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, 
                 toolbox.attr_float, n=n_assets)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=gene_mutation_rate)
toolbox.register("select", tools.selTournament, tournsize=tourn_size)
toolbox.register("evaluate", evaluate_portfolio)

stats = tools.Statistics(lambda ind: ind.fitness.values[0])
stats.register("min", np.min)

population = toolbox.population(n=pop_size)

hof = tools.HallOfFame(hof_size)


gen_best_fitness = []
for gen in range(1, n_generations + 1):
    if gen % restart_interval == 0:
        new_population = []
        for _ in range(pop_size - hof_size):
            new_individual = toolbox.individual()
            new_population.append(new_individual)
        new_population.extend(hof)
        hof.update(population)
        population = new_population
        crossover_rate = np.random.uniform(0.5, 1)
        mutation_rate = np.random.uniform(0.0, 0.5)
        gene_mutation_rate = random.uniform(0.0, 0.5)
    population, log = algorithms.eaSimple(
        population, toolbox, cxpb=crossover_rate, mutpb=mutation_rate,
        ngen=1, stats=stats, halloffame=hof, verbose=False
    )
    gen_best_fitness.append(log.select("min"))



plt.figure(figsize=(12, 6))
best_so_far = penalty
plotted_best_fitness = []
for gen_stats in gen_best_fitness:
    min_fitness = min(gen_stats)
    best_so_far = min(best_so_far, min_fitness)
    if best_so_far < penalty:
        plotted_best_fitness.append(best_so_far)
    else:
        plotted_best_fitness.append(None)

plotted_generations = list(range(len(plotted_best_fitness)))
plotted_values = [val if val is not None else np.nan for val in plotted_best_fitness]

plt.plot(plotted_generations, plotted_values, label='Best value', linewidth=2)
plt.xlabel('Generation')
plt.ylabel('Fitness (Risk)')
plt.title('Evolution of Best Fitness by Generation')
plt.legend()
plt.grid(True)
plt.savefig(f"./unit-10/output/portfolio_optimization_{instance_name}.png")

best_solution = hof[0]
print("\nBest solution:")
print("  - Weights:", [f"{weight:.3f}" for weight in best_solution])
print("  - Return:", f"{np.dot(best_solution, expected_returns):.4f}")
print("  - Risk:", f"{best_solution.fitness.values[0]:.6f}")

# Mostrar distribución de activos
print("\nAsset Distribution:")
for i, weight in enumerate(best_solution):
    print(f"  - Asset {i+1}: {weight:.1%}")