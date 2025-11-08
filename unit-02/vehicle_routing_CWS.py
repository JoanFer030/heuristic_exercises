import json
import time
import math
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

def get_saving_list(G: nx.Graph, depot_id: int = 0) -> list[tuple]:
    savings_list = []
    non_depot_nodes = [node for node in G.nodes() if node != depot_id]
    for i in range(len(non_depot_nodes)):
        for j in range(i + 1, len(non_depot_nodes)):
            node_i = non_depot_nodes[i]
            node_j = non_depot_nodes[j]
            # Calculate saving: c(depot,i) + c(depot,j) - c(i,j)
            saving = (G[depot_id][node_i]["cost"] + G[depot_id][node_j]["cost"] - G[node_i][node_j]["cost"])
            savings_list.append((saving, node_i, node_j))
    # Sort savings in descending order
    savings_list.sort(reverse=True, key=lambda x: x[0])
    return savings_list

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

def can_merge_routes(G: nx.Graph, route_i: dict, route_j: dict, node_i: int, node_j: int, 
                    vehicle_capacity: float, depot_id: int = 0) -> bool:
    # Check if routes are different
    if route_i == route_j:
        return False
    # Check capacity constraint
    if route_i["demand"] + route_j["demand"] > vehicle_capacity:
        return False
    # Check if nodes are interior (not allowed to merge interior nodes)
    if G.nodes[node_i]["is_interior"] or G.nodes[node_j]["is_interior"]:
        return False
    return True


def merge_routes(G: nx.Graph, routes: list[dict], route_i_idx: int, route_j_idx: int, 
                node_i: int, node_j: int, saving: float, depot_id: int = 0):
    route_i = routes[route_i_idx]
    route_j = routes[route_j_idx]

    seq_i = route_i["nodes"][1:-1]
    seq_j = route_j["nodes"][1:-1]
    if len(seq_i) == 0 or len(seq_j) == 0:
        return  # safety

    # Four possibilities: node_i at end/start; node_j at start/end
    if seq_i[-1] == node_i and seq_j[0] == node_j:
        merged_seq = seq_i + seq_j
    elif seq_i[0] == node_i and seq_j[-1] == node_j:
        merged_seq = seq_j + seq_i
    elif seq_i[0] == node_i and seq_j[0] == node_j:
        # reverse seq_i so node_i becomes at the end
        merged_seq = seq_i[::-1] + seq_j
    elif seq_i[-1] == node_i and seq_j[-1] == node_j:
        # reverse seq_j so node_j becomes at the beginning
        merged_seq = seq_i + seq_j[::-1]
    else:
        # nodes are not at route ends (shouldn't happen if can_merge_routes checked), abort
        return

    # Build new node list with depot endpoints
    new_nodes = [depot_id] + merged_seq + [depot_id]
    new_cost = 0.0
    new_edges = []
    for k in range(len(new_nodes) - 1):
        a = new_nodes[k]
        b = new_nodes[k + 1]
        new_cost += G[a][b]["cost"]
        new_edges.append((a, b))
    new_demand = route_i["demand"] + route_j["demand"]
    merged_route = {
        "nodes": new_nodes,
        "cost": new_cost,
        "demand": new_demand,
        "edges": new_edges
    }

    # Rebuild routes list
    new_routes = []
    for idx, r in enumerate(routes):
        if idx == route_i_idx:
            new_routes.append(merged_route)
        elif idx == route_j_idx:
            continue  # drop route_j
        else:
            new_routes.append(r)
    routes[:] = new_routes

    # Update node attributes: route index and is_interior
    for idx, r in enumerate(routes):
        for node in r["nodes"][1:-1]:
            G.nodes[node]["route"] = idx
            pos = r["nodes"].index(node)
            if pos == 1 or pos == len(r["nodes"]) - 2:
                G.nodes[node]["is_interior"] = False
            else:
                G.nodes[node]["is_interior"] = True


def cws_solution(G: nx.Graph, saving_list: list[tuple], vehicle_capacity: float, depot_id: int = 0):
    # Start with dummy solution
    routes = dummy_solution(G, vehicle_capacity, depot_id)
    
    # Process savings list
    for saving, node_i, node_j in saving_list:
        route_i_idx = G.nodes[node_i]["route"]
        route_j_idx = G.nodes[node_j]["route"]
        
        route_i = routes[route_i_idx]
        route_j = routes[route_j_idx]
        
        if can_merge_routes(G, route_i, route_j, node_i, node_j, vehicle_capacity, depot_id):
            merge_routes(G, routes, route_i_idx, route_j_idx, node_i, node_j, saving, depot_id)
    
    return routes

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
    
    plt.title(f"CWS Solution for {instance_name}\nTotal Cost: {total_cost:.2f}")
    plt.axis("equal")
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"./unit-02/output/{instance_name}_CWS.png")

#######################################################################
###                         SOLVING & RESULTS                       ###
#######################################################################
instance_name = "P-n70-k10"
depot_node = 0

veh_capacity, nodes = load_instance(instance_name)
G = generate_graph(nodes, depot_node)
saving_list = get_saving_list(G, depot_node)

dummy_sol = dummy_solution(G, veh_capacity, depot_node)
total_cost = sum(route['cost'] for route in dummy_sol)
print("Dummy Solution")
print(f"Number of routes: {len(dummy_sol)}")
print(f"Total cost: {total_cost:.2f}")

t0 = time.time()
cws_sol = cws_solution(G, saving_list, veh_capacity, depot_node)
total_cost = sum(route['cost'] for route in cws_sol)
print("\n\nClarke & Wright Savings Algoritm")
print(f"Total cost: {total_cost:.2f}")
print(f"Total execution time: {time.time() - t0} s")
print("\nDetailed routes:")
for i, route in enumerate(cws_sol):
    route_str = " -> ".join(str(node) for node in route['nodes'])
    print(f"Route {i+1}: {route_str}   | Cost: {route['cost']:.2f}")
plot_solution(G, cws_sol, instance_name, total_cost, depot_node)