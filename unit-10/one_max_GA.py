import random
import matplotlib.pyplot as plt

# GA parameters
pop_size = 100 
num_generations = 100
mutation_rate = 0.001
string_length = 20

# Variables
best_sol = None
best_value = 0
average_values = []
best_values = []

def fitness_function(sol):
    return sum(sol)

def mutate(sol):
    mutated_sol = list(sol)
    for i in range(len(mutated_sol)):
        if random.random() < mutation_rate:
            mutated_sol[i] = 1 - mutated_sol[i]
    return mutated_sol



population = [random.choices([0, 1], k=string_length) for _ in range(pop_size)]
fitness_values = [fitness_function(sol) for sol in population]
for generation in range(num_generations):
    parents = random.choices(population, weights=fitness_values, k=pop_size)
    
    children = []
    for i in range(0, pop_size, 2):
        parent1, parent2 = parents[i], parents[i + 1]
        crossover_point = random.randint(1, string_length - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        children.extend([mutate(child1), mutate(child2)])
    
    population = children
    fitness_values = [fitness_function(sol) for sol in population]
    
    best_index = fitness_values.index(max(fitness_values))
    gen_best_sol = population[best_index]
    gen_best_value = fitness_values[best_index]
    if gen_best_value > best_value:
        best_sol = population[best_index]
        best_value = gen_best_value
        print(f"Nuevo mejor valor: {best_value}")
    
    average_value = sum(fitness_values) / len(fitness_values)
    average_values.append(average_value)
    best_values.append(best_value)

print(f"Best solution: {best_sol}")
print(f"Best value: {best_value}")

generations = list(range(1, num_generations + 1))
plt.plot(generations, average_values, label="Average value")
plt.plot(generations, best_values, label="Best value")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("Evolution of Average and Best Fitness by Generation")
plt.legend()
plt.ylim(bottom=0)
plt.savefig("./unit-10/output/one_max.png")