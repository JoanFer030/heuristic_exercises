import gzip
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random
import array

def parse_tsp(instance_name: str):
    coords = []
    with gzip.open(f"./data/tsp/{instance_name}.tsp.gz", "r") as f:
        lines = [l.decode().strip() for l in f.readlines()]
        start = lines.index("NODE_COORD_SECTION")
        end = lines.index("EOF")
        for line in lines[start+1:end]:
            parts = line.split()
            x, y = float(parts[1]), float(parts[2])
            coords.append((x, y))
    return np.array(coords)

class TravelingSalesmanProblem:
    def __init__(self, name):
        """Creates an instance of a TSP"""
        self.name = name
        self.locations = []
        self.distances = []
        self.tspSize = 0
        
        # Get coordinates from parsed TSP file
        self.locations = parse_tsp(name)
        self.tspSize = len(self.locations)
        
        # Calculate distance matrix
        self.__createDistanceMatrix()
    
    def __len__(self):
        """Returns the length of the underlying TSP (number of cities)."""
        return self.tspSize
    
    def __createDistanceMatrix(self):
        """Calculates the distance matrix between all cities."""
        self.distances = [[0] * self.tspSize for _ in range(self.tspSize)]
        
        for i in range(self.tspSize):
            for j in range(i + 1, self.tspSize):
                # Calculate Euclidean distance
                distance = np.linalg.norm(self.locations[j] - self.locations[i])
                self.distances[i][j] = distance
                self.distances[j][i] = distance
    
    def getTotalDistance(self, indices):
        """Calculates the total distance of the path described by given indices."""
        if len(indices) == 0:
            return 0
        
        # Distance between last and first city
        distance = self.distances[indices[-1]][indices[0]]
        
        # Add distances between consecutive cities
        for i in range(len(indices) - 1):
            distance += self.distances[indices[i]][indices[i + 1]]
        
        return distance

def easimpleWithElitism(population, toolbox, cxpb, mutpb, ngen, stats=None,
                        halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    if halloffame is None:
        raise ValueError("halloffame parameter must not be empty!")
    halloffame.update(population)
    hof_size = len(halloffame.items) if halloffame.items else 0
    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)
    for gen in range(1, ngen + 1):
        offspring = toolbox.select(population, len(population) - hof_size)
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        offspring.extend(halloffame.items)
        halloffame.update(offspring)
        population[:] = offspring
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)
    
    return population, logbook

def solve_tsp(instance_name, 
              population_size=300, 
              max_generations=200,
              hof_size=30,
              p_crossover=0.9,
              p_mutation=0.1,
              tournament_size=2,
              random_seed=42):
    random.seed(random_seed)
    
    print(f"Solving TSP instance: {instance_name}")
    print(f"Population: {population_size}, Generations: {max_generations}")
    print(f"Elitism: {hof_size} best individuals preserved")
    tsp = TravelingSalesmanProblem(instance_name)
    
    toolbox = base.Toolbox()
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", array.array, typecode='i', 
                   fitness=creator.FitnessMin)
    toolbox.register("randomOrder", random.sample, range(len(tsp)), len(tsp))
    toolbox.register("individualCreator", tools.initIterate, 
                     creator.Individual, toolbox.randomOrder)
    toolbox.register("populationCreator", tools.initRepeat, 
                     list, toolbox.individualCreator)
    
    # Fitness function
    def tspDistance(individual):
        return tsp.getTotalDistance(individual),
    
    toolbox.register("evaluate", tspDistance)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)
    toolbox.register("mate", tools.cxOrdered) 
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1.0/len(tsp))
    population = toolbox.populationCreator(n=population_size)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    stats.register("std", np.std)
    hof = tools.HallOfFame(hof_size)
    print("\nStarting Genetic Algorithm...")
    population, logbook = easimpleWithElitism(
        population, toolbox, 
        cxpb=p_crossover, 
        mutpb=p_mutation,
        ngen=max_generations, 
        stats=stats, 
        halloffame=hof, 
        verbose=False
    )
    
    best = hof.items[0]
    best_distance = best.fitness.values[0]
    print(f"Best distance: {best_distance:.2f}")
    print(f"Best route length: {len(best)} cities")
    plot_results(tsp, best, logbook, instance_name)
    return best, best_distance, logbook

def plot_results(tsp, best_solution, logbook, instance_name):    
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    cities_x = [tsp.locations[i][0] for i in range(len(tsp))]
    cities_y = [tsp.locations[i][1] for i in range(len(tsp))]
    ax1.scatter(cities_x, cities_y, color='red', s=50, alpha=0.7, label='Cities')
    route_x = [tsp.locations[i][0] for i in best_solution]
    route_y = [tsp.locations[i][1] for i in best_solution]
    route_x.append(route_x[0])
    route_y.append(route_y[0])
    ax1.plot(route_x, route_y, 'b-', linewidth=1.5, alpha=0.7, label='Route')
    ax1.scatter(route_x[0], route_y[0], color='green', s=100, 
                marker='o', edgecolors='black', zorder=5, label='Start/End')
    
    ax1.set_title(f'TSP Solution - {instance_name}\nDistance: {best_solution.fitness.values[0]:.2f}')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"./unit-10/output/{instance_name}_route.png")
    
    fig, ax2 = plt.subplots(1, 1, figsize=(8, 6))
    min_fitness = logbook.select("min")
    avg_fitness = logbook.select("avg")
    max_fitness = logbook.select("max")
    generations = range(len(min_fitness))
    ax2.plot(generations, min_fitness, 'r-', linewidth=2, label='Best Fitness')
    ax2.plot(generations, avg_fitness, 'b-', linewidth=2, label='Average Fitness')
    ax2.plot(generations, max_fitness, 'g--', linewidth=1, alpha=0.5, label='Worst Fitness')
    ax2.fill_between(generations, min_fitness, avg_fitness, alpha=0.1, color='red')
    ax2.set_title('Genetic Algorithm Convergence')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Distance (Fitness)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"./unit-10/output/{instance_name}_convergence.png")

if __name__ == "__main__":
    import time
    names = ["berlin52", "bier127", "pr76", "fl417"]
    for name in names:
        print("Instance name:", name)
        t0 = time.time()
        best_solution, best_distance, logbook = solve_tsp(
            instance_name = name,
            population_size=300,
            max_generations=200,
            hof_size=30,
            p_crossover=0.9,
            p_mutation=0.1,
            tournament_size=2,
            random_seed=42
        )
    print(f"Total time: {time.time() - t0:.4f}s")