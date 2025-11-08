import time
import json
import math
import random
import networkx as nx
import matplotlib.pyplot as plt


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

def generate_graph(nodes: list, depot_index: int = 0) -> nx.Graph:
    G = nx.Graph()
    # Add nodes (depot and others)
    for node_id, x, y, demand in nodes:
        G.add_node(node_id, x = x, y = y, demand = demand, is_interior = False) # An interior node in not connected to depot
    # Add nodes and calculate distances for all edges
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            node_i = nodes[i]
            node_j = nodes[j]
            distance = math.sqrt((node_j[1] - node_i[1])**2 + (node_j[2] - node_i[2])**2)
            G.add_edge(node_i[0], node_j[0], weight=distance, cost=distance)
    return G

def dummy_solution(G: nx.Graph, vehicle_capacity: float, depot_id: int = 0):
    routes = []
    # Create one route per customer (from depot to customer and back)
    for node in G.nodes():
        if node != depot_id:
            # route cost must be depot->node + node->depot (symmetric here)
            roundtrip_cost = G[depot_id][node]["cost"] + G[node][depot_id]["cost"]
            route = {
                "nodes": [depot_id, node, depot_id],
                "cost": roundtrip_cost,
                "demand": G.nodes[node]["demand"],
                "edges": [(depot_id, node), (node, depot_id)]
            }
            routes.append(route)
            G.nodes[node]["route"] = len(routes) - 1
            G.nodes[node]["position_in_route"] = 1  # Between depot nodes
            G.nodes[node]["is_interior"] = False
    return routes

def random_search_solution(G, veh_capacity, depot_node, max_iterations):
    """
    Random Search heuristic for the Vehicle Routing Problem.
    It generates random permutations of customers and evaluates their cost.
    """
    customers = [n[0] for n in nodes if n[0] != depot_node]
    best_routes = None
    best_cost = float("inf")
    history = []

    # Random Search main loop
    for it in range(max_iterations):
        random.shuffle(customers)
        routes = []
        current_route = [depot_node]
        current_demand = 0
        # Build feasible routes
        for c in customers:
            demand = G.nodes[c]["demand"]
            if current_demand + demand <= veh_capacity:
                current_route.append(c)
                current_demand += demand
            else:
                # Close route
                if len(current_route) > 1:
                    current_route.append(depot_node)
                    routes.append(current_route)
                    current_route = [depot_node, c]
                    current_demand = demand
                else:
                    customers.append(c)
        current_route.append(depot_node)
        routes.append(current_route)

        # Compute total cost
        total_cost = 0
        route_dicts = []
        for r in routes:
            route_edges = []
            route_cost = 0
            for i in range(len(r) - 1):
                a, b = r[i], r[i + 1]
                route_edges.append((a, b))
                route_cost += G[a][b]["cost"]
            route_dicts.append({
                "nodes": r,
                "cost": route_cost,
                "demand": sum(G.nodes[n]["demand"] for n in r if n != depot_node),
                "edges": route_edges
            })
            total_cost += route_cost

        # Update best solution
        if total_cost < best_cost:
            best_cost = total_cost
            best_routes = route_dicts

        # Store progress every 1000 iterations
        if it % 1000 == 0:
            history.append(best_cost)

    return best_routes, history


def plot_solution(G, routes, instance_name, total_cost, depot_node=0):
    plt.figure(figsize=(12, 8))
    pos = {node: (G.nodes[node]["x"], G.nodes[node]["y"]) for node in G.nodes()}
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color="lightblue")
    nx.draw_networkx_nodes(G, pos, nodelist=[depot_node], node_size=150, node_color="red", node_shape="s")
    nx.draw_networkx_labels(G, pos, labels={depot_node: "Depot"}, font_size=10, font_weight="bold")
    labels = {node: str(node) for node in G.nodes() if node != depot_node}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=6)
    
    # Draw routes
    colors = plt.cm.Set3.colors
    for i, route in enumerate(routes):
        route_color = colors[i % len(colors)]
        nx.draw_networkx_edges(G, pos, edgelist=route["edges"], edge_color=route_color, width=2.5, alpha=0.8, label=f"Route {i+1}")
    
    plt.title(f"RS Solution for {instance_name}\nTotal Cost: {total_cost:.2f}")
    plt.axis("equal")
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"./unit-02/output/{instance_name}_RS.png")

def plot_convergence(history, dummy_cost, instance_name):
    """
    Plot the convergence of random search compared to greedy solutions
    """
    iterations = list(range(0, len(history) * 1000, 1000))
    
    plt.figure(figsize = (10, 6))
    plt.plot(iterations, history, "b-", linewidth = 2, label = f"Random Search: {max(history):.2f}")
    plt.axhline(y = dummy_cost, color = "r", linestyle = "--", linewidth = 2, label = f"Dummy Algorithm: {dummy_cost:.2f}")
    plt.xlabel("Iterations")
    plt.ylabel("Best Value Found")
    plt.title(f"Vehicle Routing Problem - {instance_name}")
    plt.xscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"./unit-02/output/{instance_name}_convergence_RS.png")

#######################################################################
###                         SOLVING & RESULTS                       ###
#######################################################################
instance_name = "M-n121-k7"
depot_node = 0
max_iterarions = 100000

veh_capacity, nodes = load_instance(instance_name)
G = generate_graph(nodes, depot_node)

dummy_sol = dummy_solution(G, veh_capacity, depot_node)
dummy_total_cost = sum(route['cost'] for route in dummy_sol)
print("Dummy Solution")
print(f"Number of routes: {len(dummy_sol)}")
print(f"Total cost: {dummy_total_cost:.2f}")

t0 = time.time()
rs_sol, history = random_search_solution(G, veh_capacity, depot_node, max_iterarions)
elapsed_time = time.time() - t0
total_cost = sum(route['cost'] for route in rs_sol)
print("\n\nRandom Search Algoritm")
print(f"Elapsed time: {elapsed_time:.4f}s")
print(f"Total cost: {total_cost:.2f}")
print("\nDetailed routes:")
for i, route in enumerate(rs_sol):
    route_str = " -> ".join(str(node) for node in route['nodes'])
    print(f"Route {i+1}: {route_str}   | Cost: {route['cost']:.2f}")
plot_solution(G, rs_sol, instance_name, total_cost, depot_node)
plot_convergence(history, dummy_total_cost, instance_name)