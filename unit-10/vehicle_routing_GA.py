import json
import time
import math
import random
import matplotlib.pyplot as plt
import numpy as np

def load_instance(instance_name: str) -> tuple[float, list[tuple]]:
    with open("./data/vrp/vehicle_capacities.json", "r") as file:
        capacities = json.load(file)
    if instance_name not in capacities:
        raise ValueError(f"Instance {instance_name} not available")
   
    with open(f"./data/vrp/{instance_name}_input_nodes.txt", "r") as file:
        nodes = []
        lines = file.readlines()
        for i, line in enumerate(lines):
            values = line.strip().split()
            if len(values) >= 3:
                x, y, demand = map(float, values[:3])
                nodes.append((i, x, y, demand))
    capacity = capacities[instance_name]
    return capacity, nodes

def euclidean_distance(node1: tuple, node2: tuple) -> float:
    return math.sqrt((node1[1] - node2[1])**2 + (node1[2] - node2[2])**2)

class VRPInstance:
    def __init__(self, instance_name: str):
        """Initialize VRP instance"""
        self.name = instance_name
        self.capacity, self.nodes = load_instance(instance_name)
        self.depot = self.nodes[0]
        self.customers = self.nodes[1:]
        self.distance_matrix = self._calculate_distance_matrix()
        self.num_customers = len(self.customers)
        self.total_demand = sum(node[3] for node in self.customers)
        print(f"\nLoaded VRP instance: {self.name}")
        print(f"  Customers: {self.num_customers}")
        print(f"  Vehicle capacity: {self.capacity}")
        print(f"  Total demand: {self.total_demand}")
        print(f"  Min vehicles required: {math.ceil(self.total_demand / self.capacity)}")
    
    def _calculate_distance_matrix(self) -> list[list[float]]:
        n = len(self.nodes)
        dist_matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                dist = euclidean_distance(self.nodes[i], self.nodes[j])
                dist_matrix[i][j] = dist
                dist_matrix[j][i] = dist
        return dist_matrix
    
    def get_distance(self, node1_id: int, node2_id: int) -> float:
        return self.distance_matrix[node1_id][node2_id]
    
    def get_node(self, node_id: int) -> tuple:
        return self.nodes[node_id]
    
    def get_demand(self, node_id: int) -> float:
        return self.nodes[node_id][3]
    
    def create_random_solution(self) -> list[int]:
        customers = list(range(1, len(self.nodes)))
        random.shuffle(customers)
        return customers
    
    def evaluate_solution(self, solution: list[int], routes: list[list[int]] = None) -> dict:
        if routes is None:
            routes = self.split_into_routes(solution)
        
        route_distances = []
        route_demands = []
        total_distance = 0.0
        feasible = True
        for route in routes:
            if not route:
                route_dist = 0.0
            else:
                route_dist = self.get_distance(0, route[0])
                for i in range(len(route) - 1):
                    route_dist += self.get_distance(route[i], route[i + 1])
                route_dist += self.get_distance(route[-1], 0)
            route_demand = sum(self.get_demand(node_id) for node_id in route)
            if route_demand > self.capacity:
                feasible = False
            route_distances.append(route_dist)
            route_demands.append(route_demand)
            total_distance += route_dist
        penalty = 0.0
        if not feasible:
            excess_demand = sum(max(0, d - self.capacity) for d in route_demands)
            penalty = excess_demand * 1000
        
        return {
            'total_distance': total_distance + penalty,
            'penalty': penalty,
            'feasible': feasible,
            'num_routes': len(routes),
            'route_distances': route_distances,
            'route_demands': route_demands,
            'routes': routes
        }
    
    def split_into_routes(self, solution: list[int]) -> list[list[int]]:
        routes = []
        current_route = []
        current_demand = 0.0
        for customer_id in solution:
            customer_demand = self.get_demand(customer_id)
            if current_demand + customer_demand > self.capacity:
                if current_route:
                    routes.append(current_route)
                current_route = [customer_id]
                current_demand = customer_demand
            else:
                current_route.append(customer_id)
                current_demand += customer_demand
        if current_route:
            routes.append(current_route)
        return routes
    
    def plot_solution(self, routes: list[list[int]], title: str) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(12, 10))
        all_x = [node[1] for node in self.nodes]
        all_y = [node[2] for node in self.nodes]
        ax.scatter(all_x[1:], all_y[1:], c='gray', s=50, alpha=0.5, label='Customers')
        depot_x, depot_y = all_x[0], all_y[0]
        ax.scatter(depot_x, depot_y, c='red', s=200, 
                  marker='s', edgecolors='black', linewidth=2, label='Depot')
        colors = plt.cm.tab10(np.linspace(0, 1, len(routes)))
        total_distance = 0
        for i, route in enumerate(routes):
            if not route:
                continue
            complete_route = [0] + route + [0]
            route_x = [self.nodes[node_id][1] for node_id in complete_route]
            route_y = [self.nodes[node_id][2] for node_id in complete_route]
            route_dist = 0
            for j in range(len(complete_route) - 1):
                route_dist += self.get_distance(complete_route[j], complete_route[j + 1])
            total_distance += route_dist
            ax.plot(route_x, route_y, '-', color=colors[i], linewidth=2, 
                   alpha=0.7, label=f'Route {i+1} ({len(route)} customers)')
        ax.set_title(f'{title}\nTotal Distance: {total_distance:.2f}')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(True, alpha=0.3)        
        plt.tight_layout()
        return fig

class VRPGeneticAlgorithm:
    def __init__(self, vrp_instance: VRPInstance):
        self.vrp = vrp_instance
        self.best_solution = None
        self.best_fitness = float('inf')
        self.best_routes = None
        
        # Default parameters
        self.params = {
            'population_size': 100,
            'generations': 300,
            'crossover_rate': 0.8,
            'mutation_rate': 0.2,
            'tournament_size': 3,
            'elite_size': 5,
            'crossover_type': 'ox',  
            'mutation_type': 'swap',
            'penalty_factor': 1000.0
        }
        self.stats = {
            'best_fitness': [],
            'avg_fitness': [],
            'best_feasible': [],
            'execution_time': 0.0
        }
    
    def set_parameters(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value
    
    def initialize_population(self, size: int) -> list[dict]:
        population = []
        for _ in range(size):
            solution = self.vrp.create_random_solution()
            routes = self.vrp.split_into_routes(solution)
            fitness = self.vrp.evaluate_solution(solution, routes)
            population.append({
                'solution': solution,
                'routes': routes,
                'fitness': fitness['total_distance'],
                'feasible': fitness['feasible'],
                'evaluation': fitness
            })
        
        return population
    
    def tournament_selection(self, population: list[dict], tournament_size: int) -> dict:
        tournament = random.sample(population, tournament_size)
        return min(tournament, key=lambda x: x['fitness'])
    
    def order_crossover(self, parent1: list[int], parent2: list[int]) -> tuple[list[int], list[int]]:
        size = len(parent1)
        point1 = random.randint(0, size - 2)
        point2 = random.randint(point1 + 1, size - 1)
        child1 = [-1] * size
        child2 = [-1] * size
        child1[point1:point2] = parent1[point1:point2]
        child2[point1:point2] = parent2[point1:point2]
        self._ox_fill(child1, parent2, point1, point2)
        self._ox_fill(child2, parent1, point1, point2)
        return child1, child2
    
    def _ox_fill(self, child: list[int], parent: list[int], start: int, end: int):
        size = len(child)
        parent_pos = end % size
        for i in range(size):
            pos = (end + i) % size
            if child[pos] == -1: 
                while parent[parent_pos % size] in child:
                    parent_pos = (parent_pos + 1) % size
                child[pos] = parent[parent_pos % size]
                parent_pos = (parent_pos + 1) % size
    
    def pmx_crossover(self, parent1: list[int], parent2: list[int]) -> tuple[list[int], list[int]]:
        size = len(parent1)
        point1 = random.randint(0, size - 2)
        point2 = random.randint(point1 + 1, size - 1)
        child1 = parent2.copy()
        child2 = parent1.copy()
        mapping1 = {}
        mapping2 = {}
        for i in range(point1, point2):
            mapping1[parent1[i]] = parent2[i]
            mapping2[parent2[i]] = parent1[i]
        self._pmx_apply_mapping(child1, mapping1, point1, point2)
        self._pmx_apply_mapping(child2, mapping2, point1, point2)
        return child1, child2
    
    def _pmx_apply_mapping(self, child: list[int], mapping: dict, start: int, end: int):
        for i in range(len(child)):
            if start <= i < end:
                continue
            value = child[i]
            while value in mapping:
                value = mapping[value]
            child[i] = value
    
    def cycle_crossover(self, parent1: list[int], parent2: list[int]) -> tuple[list[int], list[int]]:
        size = len(parent1)
        child1 = [-1] * size
        child2 = [-1] * size
        cycles = []
        visited = [False] * size
        for i in range(size):
            if not visited[i]:
                cycle = []
                current = i
                while not visited[current]:
                    visited[current] = True
                    cycle.append(current)
                    value = parent1[current]
                    current = parent2.index(value)
                cycles.append(cycle)
        for i, cycle in enumerate(cycles):
            if i % 2 == 0:
                for pos in cycle:
                    child1[pos] = parent1[pos]
                    child2[pos] = parent2[pos]
            else:
                for pos in cycle:
                    child1[pos] = parent2[pos]
                    child2[pos] = parent1[pos]
        return child1, child2
    
    def swap_mutation(self, solution: list[int]) -> list[int]:
        mutated = solution.copy()
        if random.random() < self.params['mutation_rate']:
            i, j = random.sample(range(len(mutated)), 2)
            mutated[i], mutated[j] = mutated[j], mutated[i]
        return mutated
    
    def inversion_mutation(self, solution: list[int]) -> list[int]:
        mutated = solution.copy()
        if random.random() < self.params['mutation_rate']:
            i, j = sorted(random.sample(range(len(mutated)), 2))
            mutated[i:j+1] = reversed(mutated[i:j+1])
        return mutated
    
    def scramble_mutation(self, solution: list[int]) -> list[int]:
        mutated = solution.copy()
        if random.random() < self.params['mutation_rate']:
            i, j = sorted(random.sample(range(len(mutated)), 2))
            segment = mutated[i:j+1]
            random.shuffle(segment)
            mutated[i:j+1] = segment
        return mutated
    
    def insertion_mutation(self, solution: list[int]) -> list[int]:
        mutated = solution.copy()
        if random.random() < self.params['mutation_rate']:
            i = random.randint(0, len(mutated) - 1)
            j = random.randint(0, len(mutated) - 1)
            if i != j:
                gene = mutated.pop(i)
                mutated.insert(j, gene)
        return mutated
    
    def crossover(self, parent1: list[int], parent2: list[int]) -> tuple[list[int], list[int]]:
        if random.random() > self.params['crossover_rate']:
            return parent1.copy(), parent2.copy()
        crossover_type = self.params['crossover_type'].lower()
        if crossover_type == 'pmx':
            return self.pmx_crossover(parent1, parent2)
        elif crossover_type == 'cx':
            return self.cycle_crossover(parent1, parent2)
        else: 
            return self.order_crossover(parent1, parent2)
    
    def mutate(self, solution: list[int]) -> list[int]:
        mutation_type = self.params['mutation_type'].lower()
        if mutation_type == 'inversion':
            return self.inversion_mutation(solution)
        elif mutation_type == 'scramble':
            return self.scramble_mutation(solution)
        elif mutation_type == 'insertion':
            return self.insertion_mutation(solution)
        else: 
            return self.swap_mutation(solution)
    
    def repair_solution(self, solution: list[int]) -> list[int]:
        all_customers = set(range(1, self.vrp.num_customers + 1))
        present_customers = set(solution)
        if len(present_customers) < self.vrp.num_customers:
            missing = list(all_customers - present_customers)
            random.shuffle(missing)
            seen = set()
            unique_solution = []
            for cust in solution:
                if cust not in seen:
                    unique_solution.append(cust)
                    seen.add(cust)
            unique_solution.extend(missing)
            return unique_solution[:self.vrp.num_customers]
        elif len(solution) != len(set(solution)):
            seen = set()
            repaired = []
            for cust in solution:
                if cust not in seen:
                    repaired.append(cust)
                    seen.add(cust)
            missing = list(all_customers - set(repaired))
            random.shuffle(missing)
            repaired.extend(missing)
            return repaired[:self.vrp.num_customers]
        return solution
    
    def run(self, verbose: bool = True) -> dict:
        start_time = time.time()
        params = self.params
        population = self.initialize_population(params['population_size'])
        for generation in range(params['generations']):
            population.sort(key=lambda x: x['fitness'])
            if population[0]['fitness'] < self.best_fitness:
                self.best_solution = population[0]['solution'].copy()
                self.best_fitness = population[0]['fitness']
                self.best_routes = population[0]['routes'].copy()
            fitness_values = [ind['fitness'] for ind in population]
            self.stats['best_fitness'].append(min(fitness_values))
            self.stats['avg_fitness'].append(sum(fitness_values) / len(fitness_values))
            feasible_count = sum(1 for ind in population if ind['feasible'])
            self.stats['best_feasible'].append(feasible_count)
            new_population = population[:params['elite_size']]
            # Generate new individuals
            while len(new_population) < params['population_size']:
                # Selection
                parent1 = self.tournament_selection(population, params['tournament_size'])
                parent2 = self.tournament_selection(population, params['tournament_size'])
                # Crossover
                child1_sol, child2_sol = self.crossover(
                    parent1['solution'], parent2['solution']
                )
                # Mutation
                child1_sol = self.mutate(child1_sol)
                child2_sol = self.mutate(child2_sol)
                # Repair solutions
                child1_sol = self.repair_solution(child1_sol)
                child2_sol = self.repair_solution(child2_sol)
                # Evaluate children
                child1_routes = self.vrp.split_into_routes(child1_sol)
                child1_fitness = self.vrp.evaluate_solution(child1_sol, child1_routes)
                child2_routes = self.vrp.split_into_routes(child2_sol)
                child2_fitness = self.vrp.evaluate_solution(child2_sol, child2_routes)
                # Add to new population
                new_population.append({
                    'solution': child1_sol,
                    'routes': child1_routes,
                    'fitness': child1_fitness['total_distance'],
                    'feasible': child1_fitness['feasible'],
                    'evaluation': child1_fitness
                })
                if len(new_population) < params['population_size']:
                    new_population.append({
                        'solution': child2_sol,
                        'routes': child2_routes,
                        'fitness': child2_fitness['total_distance'],
                        'feasible': child2_fitness['feasible'],
                        'evaluation': child2_fitness
                    })
            population = new_population[:params['population_size']]
            if verbose and generation % 50 == 0:
                feasible_pct = (feasible_count / params['population_size']) * 100
                print(f"Gen {generation:4d}: Best = {self.stats['best_fitness'][-1]:.2f}, Avg = {self.stats['avg_fitness'][-1]:.2f}")
        self.stats['execution_time'] = time.time() - start_time
        if self.best_routes:
            final_eval = self.vrp.evaluate_solution(self.best_solution, self.best_routes)
        else:
            self.best_routes = self.vrp.split_into_routes(self.best_solution)
            final_eval = self.vrp.evaluate_solution(self.best_solution, self.best_routes)
            self.best_fitness = final_eval['total_distance']
        print(f"Execution time: {self.stats['execution_time']:.2f} seconds")
        print(f"Best fitness: {self.best_fitness:.2f}")
        print(f"Feasible: {final_eval['feasible']}")
        print(f"Number of routes: {final_eval['num_routes']}")
        print(f"Total distance: {final_eval['total_distance']:.2f}")
        print(f"Penalty: {final_eval['penalty']:.2f}")
        return {
            'best_solution': self.best_solution,
            'best_fitness': self.best_fitness,
            'best_routes': self.best_routes,
            'evaluation': final_eval,
            'stats': self.stats
        }
    
    def plot_convergence(self) -> plt.Figure:
        """Plot convergence of the genetic algorithm"""
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
        generations = range(len(self.stats['best_fitness']))
        ax1.plot(generations, self.stats['best_fitness'], 'r-', linewidth=2, label='Best Fitness')
        ax1.plot(generations, self.stats['avg_fitness'], 'b-', linewidth=2, label='Average Fitness')
        ax1.fill_between(generations, self.stats['best_fitness'], self.stats['avg_fitness'], 
                        alpha=0.1, color='red')
        ax1.set_title(f'Genetic Algorithm Convergence - {self.vrp.name}')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Total Distance (Fitness)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig   

def plot_parameter_comparison(results: list[dict]):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    names = [r['name'] for r in results]
    fitness_values = [r['best_fitness'] for r in results]
    times = [r['execution_time'] for r in results]
    colors = plt.cm.tab10(range(len(results)))
    bars1 = ax1.bar(names, fitness_values, color=colors)
    ax1.set_title('Best Fitness Comparison')
    ax1.set_ylabel('Total Distance')
    ax1.set_xlabel('Configuration')
    ax1.tick_params(axis='x', rotation=45)
    for bar, value in zip(bars1, fitness_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}', ha='center', va='bottom')
    bars2 = ax2.bar(names, times, color=colors)
    ax2.set_title('Execution Time Comparison')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_xlabel('Configuration')
    ax2.tick_params(axis='x', rotation=45)
    for bar, value in zip(bars2, times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}s', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(f"./unit-10/output/{instance_name}_parameters.png")

def run_one_instance(instance_name: str):
    vrp = VRPInstance(instance_name)
    ga = VRPGeneticAlgorithm(vrp)
    ga.set_parameters(
        population_size=100,
        generations=300,
        crossover_rate=0.8,
        mutation_rate=0.2,
        tournament_size=3,
        elite_size=5,
        crossover_type='ox',
        mutation_type='swap'
    )
    results = ga.run(verbose=True)

    fig1 = ga.plot_convergence()
    plt.savefig(f"./unit-10/output/{instance_name}_convergence.png")
    fig2 = vrp.plot_solution(results['best_routes'], 
                           title=f"Best Solution - Distance: {results['best_fitness']:.2f}")
    plt.savefig(f"./unit-10/output/{instance_name}_routes.png")

def parameter_study(instance_name: str = "P-n70-k10"):
    vrp = VRPInstance(instance_name)
    param_combinations = [
        {
            'name': 'Default',
            'population_size': 100,
            'generations': 200,
            'crossover_rate': 0.8,
            'mutation_rate': 0.1,
            'crossover_type': 'ox',
            'mutation_type': 'swap'
        },
        {
            'name': 'High Mutation',
            'population_size': 100,
            'generations': 200,
            'crossover_rate': 0.7,
            'mutation_rate': 0.3,
            'crossover_type': 'ox',
            'mutation_type': 'swap'
        },
        {
            'name': 'Large Population',
            'population_size': 200,
            'generations': 150,
            'crossover_rate': 0.8,
            'mutation_rate': 0.1,
            'crossover_type': 'ox',
            'mutation_type': 'swap'
        },
        {
            'name': 'PMX Crossover',
            'population_size': 100,
            'generations': 200,
            'crossover_rate': 0.8,
            'mutation_rate': 0.2,
            'crossover_type': 'pmx',
            'mutation_type': 'swap'
        },
        {
            'name': 'Inversion Mutation',
            'population_size': 100,
            'generations': 200,
            'crossover_rate': 0.8,
            'mutation_rate': 0.2,
            'crossover_type': 'ox',
            'mutation_type': 'inversion'
        }
    ]
    
    results = []
    for params in param_combinations:
        print(f"\n{'='*60}")
        print(f"Testing: {params['name']}")
        print(f"{'='*60}")
        random.seed(42)
        ga = VRPGeneticAlgorithm(vrp)
        ga.set_parameters(**{k: v for k, v in params.items() if k != 'name'})
        result = ga.run(verbose=False)
        results.append({
            'name': params['name'],
            'best_fitness': result['best_fitness'],
            'execution_time': result['stats']['execution_time'],
            'feasible': result['evaluation']['feasible'],
            'num_routes': result['evaluation']['num_routes'],
            'solution': result['best_solution']
        })
        
        print(f"  Best Fitness: {result['best_fitness']:.2f}")
        print(f"  Execution Time: {result['stats']['execution_time']:.2f}s")
        print(f"  Number of Routes: {result['evaluation']['num_routes']}")
    plot_parameter_comparison(results)
    return results


def crossover_comparison(instance_name: str):
    """Compare different crossover operators"""
    vrp = VRPInstance(instance_name)
    
    crossover_types = ['ox', 'pmx', 'cx']
    results = []
    
    random.seed(42)
    for cx_type in crossover_types:
        print(f"\nTesting Crossover: {cx_type}")
        ga = VRPGeneticAlgorithm(vrp)
        ga.set_parameters(
            population_size=100,
            generations=200,
            crossover_rate=0.8,
            mutation_rate=0.2,
            crossover_type=cx_type,
            mutation_type='swap'
        )
        result = ga.run(verbose=False)
        results.append({
            'crossover': cx_type,
            'best_fitness': result['best_fitness'],
            'best_history': result['stats']['best_fitness'],
            'avg_history': result['stats']['avg_fitness']
        })
    plt.figure(figsize=(10, 6))
    for result in results:
        plt.plot(result['best_history'], 
                label=f'{result["crossover"]} - {result["best_fitness"]:.1f}')
    plt.title('Crossover Operator Comparison')
    plt.xlabel('Generation')
    plt.ylabel('Total Distance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"./unit-10/output/{instance_name}_crossover.png")
    return results

if __name__ == "__main__":
    instance_name = "P-n70-k10"

    run_one_instance(instance_name)
    parameter_study(instance_name)
    crossover_comparison(instance_name)