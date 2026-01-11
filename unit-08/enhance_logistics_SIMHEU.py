import json
import numpy as np
import random
import math
from copy import deepcopy
from dataclasses import dataclass
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

@dataclass
class Node:
    id: int
    x: float
    y: float
    deterministic_demand: float
    storage_capacity: float = 0.0
    initial_stock: float = 0.0
    
    def distance_to(self, other: 'Node') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

class IRPSolution:
    def __init__(self, num_nodes: int, num_periods: int):
        self.refill_matrix = np.zeros((num_nodes, num_periods))  # q_ip
        self.routes = {}  # Diccionario: periodo -> lista de rutas
        self.total_cost = 0.0
        self.inventory_cost = 0.0
        self.routing_cost = 0.0
        
    def copy(self):
        """Crea una copia de la solución"""
        new_solution = IRPSolution(self.refill_matrix.shape[0], self.refill_matrix.shape[1])
        new_solution.refill_matrix = self.refill_matrix.copy()
        new_solution.routes = deepcopy(self.routes)
        new_solution.total_cost = self.total_cost
        new_solution.inventory_cost = self.inventory_cost
        new_solution.routing_cost = self.routing_cost
        return new_solution

class SimheuristicIRP:
    def __init__(self, 
                 capacity: float, 
                 nodes: list[tuple],
                 holding_cost: float = 0.25,
                 stockout_multiplier: float = 2.0,
                 num_vehicles: int = 5,
                 planning_horizon: int = 3,
                 gamma: float = 0.05,
                 variance_level: float = 0.5):
        
        self.capacity = capacity
        self.holding_cost = holding_cost
        self.stockout_multiplier = stockout_multiplier
        self.num_vehicles = num_vehicles
        self.planning_horizon = planning_horizon
        self.gamma = gamma
        self.variance_level = variance_level
        self.cost_history = []
        self.best_cost_history = []
        self.elite_costs_history = []
        self.depot = Node(id=0, x=nodes[0][1], y=nodes[0][2], 
                         deterministic_demand=0.0)
        self.customers = []
        for i, (node_id, x, y, demand) in enumerate(nodes):
            if i == 0:
                continue 
            storage_cap = max(demand * 10, demand * 2) 
            self.customers.append(
                Node(id=i, x=x, y=y, 
                    deterministic_demand=demand * gamma,
                    storage_capacity=storage_cap,
                    initial_stock=storage_cap * 0.5)
            )
        self.all_nodes = [self.depot] + self.customers
        self.dist_matrix = self._calculate_distance_matrix()
        self.num_simulation_runs = 30
        self.num_iterations = 100
        self.num_elite_solutions = 5
        self.refinement_runs = 100
        
    def _calculate_distance_matrix(self) -> np.ndarray:
        n = len(self.all_nodes)
        dist_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist_matrix[i][j] = self.all_nodes[i].distance_to(self.all_nodes[j])
                    
        return dist_matrix
    
    def _generate_stochastic_demand(self, customer: Node, period: int) -> float:
        mean_demand = customer.deterministic_demand
        
        if mean_demand <= 0:
            return 0.0
        sigma = self.variance_level * 0.5
        mu = math.log(mean_demand) - 0.5 * sigma**2
        demand = np.random.lognormal(mu, sigma)
        demand = max(0, min(demand, customer.storage_capacity))
        
        return demand
    
    def solve_vrp(self, delivery_quantities: list[float]) -> tuple[list[list[int]], float]:
        nodes_to_visit = [i for i, qty in enumerate(delivery_quantities, start=1) if qty > 0]
        if not nodes_to_visit:
            return [], 0.0
        routes = [[0, node, 0] for node in nodes_to_visit]
        savings = []
        for i in range(len(nodes_to_visit)):
            for j in range(i + 1, len(nodes_to_visit)):
                node_i = nodes_to_visit[i]
                node_j = nodes_to_visit[j]
                saving = (self.dist_matrix[0][node_i] + 
                         self.dist_matrix[0][node_j] - 
                         self.dist_matrix[node_i][node_j])
                
                if saving > 0:
                    savings.append((saving, i, j))
        savings.sort(reverse=True, key=lambda x: x[0])
        for saving, i, j in savings:
            route_i = self._find_route_containing(routes, nodes_to_visit[i])
            route_j = self._find_route_containing(routes, nodes_to_visit[j])
            
            if route_i is None or route_j is None or route_i == route_j:
                continue
            if self._can_combine_routes(route_i, route_j, delivery_quantities, nodes_to_visit[i], nodes_to_visit[j]):
                combined_route = self._combine_routes(route_i, route_j, nodes_to_visit[i], nodes_to_visit[j])
                routes.remove(route_i)
                routes.remove(route_j)
                routes.append(combined_route)
        if len(routes) > self.num_vehicles:
            routes.sort(key=lambda r: sum(delivery_quantities[node-1] for node in r[1:-1]), reverse=True)
            while len(routes) > self.num_vehicles:
                route1 = routes.pop()
                route2 = routes.pop()
                combined = self._merge_routes(route1, route2)
                routes.append(combined)
        total_distance = 0.0
        for route in routes:
            for k in range(len(route) - 1):
                total_distance += self.dist_matrix[route[k]][route[k + 1]]
        
        return routes, total_distance
    
    def _find_route_containing(self, routes: list[list[int]], node: int) -> list[int]:
        for route in routes:
            if node in route:
                return route
        return None
    
    def _can_combine_routes(self, route1: list[int], route2: list[int], 
                           delivery_quantities: list[float], node_i: int, node_j: int) -> bool:
        total_demand = 0
        for node in route1[1:-1]:  # Excluir depósitos
            total_demand += delivery_quantities[node - 1]
        for node in route2[1:-1]:
            total_demand += delivery_quantities[node - 1]
        
        return total_demand <= self.capacity
    
    def _combine_routes(self, route1: list[int], route2: list[int], 
                       node_i: int, node_j: int) -> list[int]:
        # Encontrar posiciones de los nodos
        try:
            pos_i = route1.index(node_i)
            pos_j = route2.index(node_j)
            
            # Combinar dependiendo de las posiciones
            if pos_i == 1 and route2[-2] == node_j:
                # Conectar el final de route2 con el inicio de route1
                combined = route2[:-1] + route1[1:]
            elif route1[-2] == node_i and pos_j == 1:
                # Conectar el final de route1 con el inicio de route2
                combined = route1[:-1] + route2[1:]
            elif pos_i == 1 and pos_j == 1:
                # Invertir route2 y conectar
                combined = [0] + route2[1:-1][::-1] + route1[1:]
            else:
                # Combinación simple
                combined = [0] + route1[1:-1] + route2[1:-1] + [0]
            
            return combined
        except ValueError:
            # En caso de error, hacer combinación simple
            return [0] + route1[1:-1] + route2[1:-1] + [0]
    
    def _merge_routes(self, route1: list[int], route2: list[int]) -> list[int]:
        return [0] + route1[1:-1] + route2[1:-1] + [0]
    
    def evaluate_solution(self, solution: IRPSolution, 
                         demand_realization: np.ndarray = None,
                         use_simulation: bool = False) -> float:
        total_cost = 0.0
        total_inventory = 0.0
        total_routing = 0.0
        
        if use_simulation and demand_realization is None:
            for _ in range(self.num_simulation_runs):
                inv_cost, rout_cost = self._evaluate_single_scenario(solution)
                total_inventory += inv_cost
                total_routing += rout_cost
                total_cost += inv_cost + rout_cost
            
            solution.inventory_cost = total_inventory / self.num_simulation_runs
            solution.routing_cost = total_routing / self.num_simulation_runs
            return total_cost / self.num_simulation_runs
        else:
            inv_cost, rout_cost = self._evaluate_single_scenario(solution, demand_realization)
            solution.inventory_cost = inv_cost
            solution.routing_cost = rout_cost
            return inv_cost + rout_cost
    
    def _evaluate_single_scenario(self, solution: IRPSolution, 
                                 demand_realization: np.ndarray = None) -> tuple[float, float]:
        n_customers = len(self.customers)
        current_stock = np.array([c.initial_stock for c in self.customers])
        
        total_inventory_cost = 0.0
        total_routing_cost = 0.0
        
        for period in range(self.planning_horizon):
            refill_quantities = solution.refill_matrix[:, period]
            if demand_realization is None:
                demands = np.array([self._generate_stochastic_demand(c, period) 
                                  for c in self.customers])
            else:
                demands = demand_realization[:, period]
            stock_after_refill = current_stock + refill_quantities
            for i in range(n_customers):
                if stock_after_refill[i] >= demands[i]:
                    surplus = stock_after_refill[i] - demands[i]
                    total_inventory_cost += surplus * self.holding_cost
                    current_stock[i] = surplus
                else:
                    stockout_qty = demands[i] - stock_after_refill[i]
                    emergency_cost = 2 * self.dist_matrix[0][i+1] * stockout_qty
                    total_inventory_cost += emergency_cost
                    current_stock[i] = 0
            routes, routing_cost = self.solve_vrp(refill_quantities.tolist())
            total_routing_cost += routing_cost
            if period in solution.routes:
                solution.routes[period] = routes
            else:
                solution.routes[period] = routes
        
        return total_inventory_cost, total_routing_cost
    
    def generate_initial_solution(self) -> IRPSolution:
        """Genera una solución inicial homogénea"""
        n_customers = len(self.customers)
        solution = IRPSolution(n_customers, self.planning_horizon)
        for i, customer in enumerate(self.customers):
            for period in range(self.planning_horizon):
                solution.refill_matrix[i, period] = customer.deterministic_demand
        solution.total_cost = self.evaluate_solution(solution, use_simulation=True)
        return solution
    
    def biased_randomization(self, base_solution: IRPSolution, 
                           bias_strength: float = 0.3) -> IRPSolution:
        new_solution = base_solution.copy()
        n_customers, n_periods = new_solution.refill_matrix.shape
        
        for i in range(n_customers):
            customer = self.customers[i]
            for p in range(n_periods):
                current_value = new_solution.refill_matrix[i, p]
                lower_bound = 0
                upper_bound = min(customer.storage_capacity, 
                                 customer.storage_capacity * 0.8)
                if upper_bound <= lower_bound:
                    new_solution.refill_matrix[i, p] = 0
                    continue
                    
                if random.random() < 0.7:
                    mode = max(lower_bound, min(upper_bound, current_value))
                    new_value = np.random.triangular(lower_bound, mode, upper_bound)
                else:
                    new_value = np.random.uniform(lower_bound, upper_bound)
                
                new_solution.refill_matrix[i, p] = new_value
        return new_solution
    
    def multi_start_search(self, initial_solution: IRPSolution, 
                          num_starts: int = 50) -> tuple[IRPSolution, list[IRPSolution]]:
        best_solution = initial_solution
        elite_solutions = [initial_solution.copy()]
        for iteration in range(num_starts):
            if iteration < len(elite_solutions):
                base = elite_solutions[iteration % len(elite_solutions)]
            else:
                base = random.choice(elite_solutions)
            new_solution = self.biased_randomization(base)
            new_solution.total_cost = self.evaluate_solution(new_solution, use_simulation=True)
            self.cost_history.append(new_solution.total_cost)
            elite_solutions.append(new_solution)
            elite_solutions.sort(key=lambda s: s.total_cost)
            elite_solutions = elite_solutions[:self.num_elite_solutions]
            if new_solution.total_cost < best_solution.total_cost:
                best_solution = new_solution
            self.best_cost_history.append(best_solution.total_cost)
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}: Best cost = {best_solution.total_cost:.2f}")
        self.elite_costs_history = [sol.total_cost for sol in elite_solutions]
        
        return best_solution, elite_solutions
    
    def refinement_stage(self, elite_solutions: list[IRPSolution]) -> IRPSolution:
        original_runs = self.num_simulation_runs
        self.num_simulation_runs = self.refinement_runs
        for solution in elite_solutions:
            solution.total_cost = self.evaluate_solution(solution, use_simulation=True)
        elite_solutions.sort(key=lambda s: s.total_cost)
        self.num_simulation_runs = original_runs
        return elite_solutions[0]
    
    def plot_cost_convergence(self):
        fig, ax = plt.subplots(1, 1, figsize=(7, 3))
        ax.plot(range(1, len(self.cost_history) + 1), self.cost_history, 
                    'o-', alpha=0.5, label='Current cost', markersize=4)
        ax.plot(range(1, len(self.best_cost_history) + 1), self.best_cost_history, 
                    'r-', linewidth=2, label='Best cost')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Total cost')
        ax.set_title('Evolution of Average and Best Cost by Iteration')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    
    def plot_routes(self, solution: IRPSolution, period: int = 0):
        if period not in solution.routes or not solution.routes[period]:
            print(f"No hay rutas para el período {period}")
            return None
        
        routes = solution.routes[period]
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, len(routes)))
        depot = self.depot
        ax.plot(depot.x, depot.y, 'ks', markersize=15, label='Depot', markeredgewidth=2)
        for customer in self.customers:
            ax.plot(customer.x, customer.y, 'bo', markersize=8, alpha=0.7)
            ax.text(customer.x, customer.y + 0.5, f'{customer.id}', 
                   fontsize=9, ha='center', va='bottom')
        for idx, route in enumerate(routes):
            color = colors[idx % len(colors)]
            route_points = [self.all_nodes[node] for node in route]
            for i in range(len(route_points) - 1):
                start = route_points[i]
                end = route_points[i + 1]
                ax.plot([start.x, end.x], [start.y, end.y], 
                       color=color, linewidth=2, alpha=0.8)
                mid_x = (start.x + end.x) / 2
                mid_y = (start.y + end.y) / 2
                dx = end.x - start.x
                dy = end.y - start.y
                ax.arrow(mid_x, mid_y, dx * 0.1, dy * 0.1, 
                        head_width=0.5, head_length=0.5, 
                        fc=color, ec=color, alpha=0.8)
            route_x = [p.x for p in route_points]
            route_y = [p.y for p in route_points]
            ax.plot(route_x, route_y, 'o', color=color, markersize=6, 
                   label=f'Route {idx+1}')
        ax.set_xlabel('Coord X')
        ax.set_ylabel('Coord Y')
        ax.set_title(f'Routes - Period {period + 1}')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        return fig
    
    def plot_demand_distribution(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        n_samples = 100
        demands_samples = []
        for customer in self.customers[:5]:
            customer_demands = []
            for _ in range(n_samples):
                demand = self._generate_stochastic_demand(customer, 0)
                customer_demands.append(demand)
            demands_samples.append(customer_demands)
        for i, customer_demands in enumerate(demands_samples):
            axes[0].hist(customer_demands, bins=30, alpha=0.5, 
                        label=f'Customer {self.customers[i].id}', density=True)
        axes[0].set_xlabel('Demand')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Demand Distribution by Customers')
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)
        box_data = demands_samples[:min(5, len(demands_samples))]
        customer_labels = [f'C{self.customers[i].id}' for i in range(len(box_data))]
        bp = axes[1].boxplot(box_data, labels=customer_labels, patch_artist=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(box_data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        axes[1].set_xlabel('Customer')
        axes[1].set_ylabel('Demand')
        axes[1].set_title('Demand Distribution by Customers')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def solve(self, plot_results: bool = True) -> IRPSolution:
        initial_solution = self.generate_initial_solution()
        print(f"Initial cost: {initial_solution.total_cost:.2f}")

        best_solution, elite_solutions = self.multi_start_search(initial_solution)
        final_solution = self.refinement_stage(elite_solutions)
        print(f"Final cost: {final_solution.total_cost:.2f}")
        
        # Calcular estadísticas
        total_refill = np.sum(final_solution.refill_matrix)
        avg_refill_per_customer = np.mean(final_solution.refill_matrix)
        print(f"Total replenishment quantity: {total_refill:.2f}")
        print(f"Average replenishment per customer: {avg_refill_per_customer:.2f}")
        print(f"Inventory cost: {final_solution.inventory_cost:.2f}")
        print(f"Routing cost: {final_solution.routing_cost:.2f}")
        
        improvement = ((initial_solution.total_cost - final_solution.total_cost) / 
                      initial_solution.total_cost * 100)
        print(f"Saving: {initial_solution.total_cost - final_solution.total_cost:.2f} ({improvement:.1f}%)")
        
        convergence_fig = self.plot_cost_convergence()
        convergence_fig.savefig('./unit-08/output/irp_convergence.png', dpi=150, bbox_inches='tight')
        routes_fig = self.plot_routes(final_solution, 0)
        routes_fig.savefig('./unit-08/output/irp_routes.png', dpi=150, bbox_inches='tight')            
        demand_fig = self.plot_demand_distribution()
        demand_fig.savefig('./unit-08/output/irp_demand_dist.png', dpi=150, bbox_inches='tight')
        return final_solution, initial_solution


def load_instance(instance_name):
    with open("./data/vrp/vehicle_capacities.json", "r") as file:
        capacities = json.loads(file.read())
    if instance_name not in capacities:
        raise ValueError("Instance not available")
    
    with open(f"./data/vrp/{instance_name}_input_nodes.txt", "r") as file:
        nodes = []
        lines = file.readlines()
        for i, line in enumerate(lines):
            x, y, demand = [float(value.strip()) for value in line.split("\t")]
            nodes.append((i, x, y, demand))

    capacity = capacities.get(instance_name)

    return capacity, nodes


def main(instance_name):
    np.random.seed(42)
    random.seed(42)
    
    capacity, nodes = load_instance(instance_name)
    simheuristic = SimheuristicIRP(
        capacity=capacity,
        nodes=nodes,
        holding_cost=0.5,
        planning_horizon=5,  # 5 períodos para mejor visualización
        gamma=1,
        variance_level=2,
        num_vehicles=5
    )
    
    final_solution, initial_solution = simheuristic.solve(plot_results=True)
    
    return final_solution, initial_solution, simheuristic

if __name__ == "__main__":
    # Ejecutar algoritmo principal
    instance_name = "A-n32-k5"
    final_solution, initial_solution, simheuristic = main(instance_name)