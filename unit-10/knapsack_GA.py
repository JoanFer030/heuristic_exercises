import random
import matplotlib.pyplot as plt

################################
## EXAMPLE DATA
################################
example_max_capacity = 400
example_data = [
    {'weight': 9, 'value': 150},  
    {'weight': 13, 'value': 35},  
    {'weight': 153, 'value': 200},
    {'weight': 50, 'value': 160}, 
    {'weight': 15, 'value': 60},  
    {'weight': 68, 'value': 45},  
    {'weight': 27, 'value': 60},  
    {'weight': 39, 'value': 40},  
    {'weight': 23, 'value': 30},  
    {'weight': 52, 'value': 10},  
    {'weight': 11, 'value': 70},  
    {'weight': 32, 'value': 30},  
    {'weight': 24, 'value': 15},  
    {'weight': 48, 'value': 10},  
    {'weight': 73, 'value': 40},  
    {'weight': 42, 'value': 70},  
    {'weight': 43, 'value': 75},  
    {'weight': 22, 'value': 80},  
    {'weight': 7, 'value': 20},   
    {'weight': 18, 'value': 12},  
    {'weight': 4, 'value': 50},   
    {'weight': 30, 'value': 10}   
]

##################################
## RANDOM DATA
##################################
def generate_knapsack_problem(n_items, capacity_ratio):
    weights = [random.randint(1, 200) for _ in range(n_items)]
    values = [random.randint(1, 200) for _ in range(n_items)]
    items = [{"weight": w, "value": v} for w, v in zip(weights, values)]

    total_weight = sum(weights)
    capacity = int(total_weight * capacity_ratio)    
    return items, capacity

###############################
## FUNCTIONS
###############################
def selection(population, fitness_values, k, isRoulette):
    if isRoulette:
        parents = random.choices(population, weights=fitness_values, k=k)
        return parents
    else:
        parents = []
        for _ in range(k):
            tournament_candidates = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_values[i] for i in tournament_candidates]
            winner_index = tournament_candidates[tournament_fitness.index(max(tournament_fitness))]
            parents.append(population[winner_index])
        return parents

def crossover(parent1, parent2):
    if isOnePoint: 
        split_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:split_point] + parent2[split_point:]
        child2 = parent2[:split_point] + parent1[split_point:]
        return child1, child2
    else:  # cruce de dos puntos
        point1 = random.randint(1, len(parent1) - 1)
        point2 = random.randint(point1, len(parent1) - 1)
        child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
        return child1, child2

def mutate(chromosome):
    mutated_chromosome = list(chromosome)
    for i in range(len(mutated_chromosome)):
        if random.random() < mutation_rate:
            mutated_chromosome[i] = 1 - mutated_chromosome[i]
    return mutated_chromosome

###############################
## CODE
###############################
n_items = 20
random_items, random_max_capacity = generate_knapsack_problem(n_items, 0.25)

max_capacity = random_max_capacity
items = random_items

population_size = 50
num_generations = 200
mutation_rate = 0.1 
isRoulette = False  
tournament_size = 3 
isOnePoint = False  
elite_count = int(0.05 * population_size)

best_sol = None
best_value = 0 
average_values = []
best_values = []  
population = [random.choices([0, 1], k=len(items)) for _ in range(population_size)]
fitness_values = []
for sol in population:
    total_weight = sum(items[i]['weight'] for i, bit in enumerate(sol) if bit)
    if total_weight > max_capacity:  # solución inviable
        fitness_values.append(0)
    else:
        fitness_values.append(sum(items[i]['value'] for i, bit in enumerate(sol) if bit))


for generation in range(num_generations):
    elite_indices = sorted(range(len(fitness_values)), 
                          key=lambda i: fitness_values[i], reverse=True)[:elite_count]
    elite_solutions = [population[i] for i in elite_indices]
    non_elite_parents = selection(population, fitness_values, 
                                  population_size - elite_count, isRoulette)
    parents = elite_solutions + non_elite_parents
    children = elite_solutions.copy()
    for i in range(0, population_size - elite_count, 2):
        child1, child2 = crossover(parents[i], parents[i + 1])
        children.append(mutate(child1))
        children.append(mutate(child2))
    population = children
    fitness_values = []
    for sol in population:
        total_weight = sum(items[i]['weight'] for i, bit in enumerate(sol) if bit)
        if total_weight > max_capacity:
            fitness_values.append(0)
        else:
            fitness_values.append(sum(items[i]['value'] for i, bit in enumerate(sol) if bit))
    
    best_index = fitness_values.index(max(fitness_values))
    gen_best_sol = population[best_index]
    gen_best_value = fitness_values[best_index]
    
    if gen_best_value > best_value:
        best_sol = population[best_index]
        best_value = gen_best_value
    average_value = sum(fitness_values) / len(fitness_values)
    average_values.append(average_value)
    best_values.append(best_value)

print(f"Max capacity: {max_capacity}")

print(f"\nBest global solution: {best_sol}")
print(f"Best global value: {best_value}")
total_weight = sum(items[i]['weight'] for i, bit in enumerate(best_sol) if bit)
print(f"Total weight: {total_weight}")

generations = list(range(1, num_generations + 1))
plt.figure(figsize=(10, 6))
plt.plot(generations, average_values, label="Average value")
plt.plot(generations, best_values, label="Best value")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("Evolution of Average and Best Fitness by Generation")
plt.legend()
plt.grid(True)
plt.ylim(bottom=0)
plt.savefig(f"./unit-10/output/knapsack.png")