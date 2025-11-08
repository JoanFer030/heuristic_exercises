import time
import math
import operator
import networkx as nx
import matplotlib.pyplot as plt


def load_instance(instance_name: str):
    with open(f"./data/top/{instance_name}.txt", "r") as file:
        lines = file.readlines()
        fleet_size = int(lines[1].strip().split(";")[-1])
        route_max_cost = float(lines[2].strip().split(";")[-1])
        nodes = []
        for i in range(3, len(lines)):
            data = [float(x.strip()) for x in lines[i].split(";")]
            node_id = i - 3
            nodes.append({"id": node_id, "x": data[0], "y": data[1], "demand": data[2]})
    return fleet_size, route_max_cost, nodes

def generate_graph(nodes: list):
    D = nx.DiGraph()
    # Add nodes
    for node in nodes:
        D.add_node(node["id"], x = node["x"], y = node["y"], demand = node["demand"],
                   inRoute = None, isLinkedToStart = False, isLinkedToFinish = False)
    start_id = nodes[0]["id"]
    finish_id = nodes[-1]["id"]
    # Create start->node and node->finish edges for non-depot nodes
    for node in nodes[1:-1]:
        nid = node["id"]
        cost_sn = math.hypot(node["x"] - nodes[0]["x"], node["y"] - nodes[0]["y"])
        cost_nf = math.hypot(node["x"] - nodes[-1]["x"], node["y"] - nodes[-1]["y"])
        D.add_edge(start_id, nid, cost=cost_sn, savings=0.0, efficiency=0.0, inv=(nid, start_id))
        D.add_edge(nid, finish_id, cost=cost_nf, savings=0.0, efficiency=0.0, inv=(finish_id, nid))
    # Create directed edges between customers (both directions) and compute efficiency
    customer_ids = [n["id"] for n in nodes[1:-1]]
    for i_idx in range(len(customer_ids)):
        i = customer_ids[i_idx]
        xi, yi = D.nodes[i]["x"], D.nodes[i]["y"]
        for j_idx in range(i_idx + 1, len(customer_ids)):
            j = customer_ids[j_idx]
            xj, yj = D.nodes[j]["x"], D.nodes[j]["y"]
            dist = math.hypot(xj - xi, yj - yi)
            # add both directed edges
            D.add_edge(i, j, cost=dist, inv=(j, i))
            D.add_edge(j, i, cost=dist, inv=(i, j))
            # compute savings/efficiency for i->j
            cost_i_finish = D.edges[i, nodes[-1]["id"]]["cost"]
            cost_start_j = D.edges[nodes[0]["id"], j]["cost"]
            savings_ij = cost_i_finish + cost_start_j - dist
            edge_reward = D.nodes[i]["demand"] + D.nodes[j]["demand"]
            eff_ij = alpha * savings_ij + (1 - alpha) * edge_reward
            D.edges[i, j].update({"savings": savings_ij, "efficiency": eff_ij})
            # compute savings/efficiency for j->i
            cost_j_finish = D.edges[j, nodes[-1]["id"]]["cost"]
            cost_start_i = D.edges[nodes[0]["id"], i]["cost"]
            savings_ji = cost_j_finish + cost_start_i - dist
            eff_ji = alpha * savings_ji + (1 - alpha) * edge_reward
            D.edges[j, i].update({"savings": savings_ji, "efficiency": eff_ji})
    return D

def dummy_solution(D: nx.DiGraph):
    sol = {"routes": [], "cost": 0.0, "demand": 0.0}
    start_id = list(D.nodes)[0]
    finish_id = list(D.nodes)[-1]
    # customers are nodes excluding start and finish
    for n in list(D.nodes):
        if n != start_id and n != finish_id:
            cost_sn = D.edges[start_id, n]["cost"]
            cost_nf = D.edges[n, finish_id]["cost"]
            route = {
                "edges": [(start_id, n), (n, finish_id)],
                "cost": cost_sn + cost_nf,
                "demand": D.nodes[n]["demand"]
            }
            sol["routes"].append(route)
            D.nodes[n]["isLinkedToStart"] = True
            D.nodes[n]["isLinkedToFinish"] = True
            D.nodes[n]["inRoute"] = route
            sol["cost"] += route["cost"]
            sol["demand"] += route["demand"]
    return sol

def merge_routes(i_id: int, j_id: int, ij_attr: dict, D: nx.DiGraph, sol: dict):
    start_id = list(D.nodes)[0]
    finish_id = list(D.nodes)[-1]
    i_route = D.nodes[i_id]["inRoute"]
    j_route = D.nodes[j_id]["inRoute"]
    i_edge_finish = (i_id, finish_id)
    if i_edge_finish in i_route["edges"]:
        i_route["edges"].remove(i_edge_finish)
        i_route["cost"] -= D.edges[i_edge_finish]["cost"]
        D.nodes[i_id]["isLinkedToFinish"] = False
    start_j = (start_id, j_id)
    if start_j in j_route["edges"]:
        j_route["edges"].remove(start_j)
        j_route["cost"] -= D.edges[start_j]["cost"]
        D.nodes[j_id]["isLinkedToStart"] = False
    i_route["edges"].append((i_id, j_id))
    i_route["cost"] += ij_attr["cost"]
    i_route["demand"] += D.nodes[j_id]["demand"]
    D.nodes[j_id]["inRoute"] = i_route
    for edge in list(j_route["edges"]):
        i_route["edges"].append(edge)
        i_route["cost"] += D.edges[edge]["cost"]
        end_node = edge[1]
        i_route["demand"] += D.nodes[end_node]["demand"]
        D.nodes[end_node]["inRoute"] = i_route
    sol["cost"] -= ij_attr["savings"]
    if j_route in sol["routes"]:
        sol["routes"].remove(j_route) 

def pjs_solution(D: nx.DiGraph, fleet_size: int, route_max_cost: float):
    # Build efficiency list
    efficiency_list = []
    start_id = list(D.nodes)[0]
    finish_id = list(D.nodes)[-1]
    customer_ids = [n for n in D.nodes if n != start_id and n != finish_id]
    for i in customer_ids:
        for j in customer_ids:
            if i == j:
                continue
            if D.has_edge(i, j) and "efficiency" in D.edges[i, j]:
                efficiency_list.append((i, j))
    efficiency_list.sort(key=lambda e: D.edges[e]["efficiency"], reverse=True)
    sol = dummy_solution(D)
    # Iterative merging
    while efficiency_list:
        i_id, j_id = efficiency_list.pop(0)
        if not D.has_edge(i_id, j_id):
            continue
        ij_attr = D.edges[i_id, j_id]
        i_route = D.nodes[i_id]["inRoute"]
        j_route = D.nodes[j_id]["inRoute"]
        if i_route is None or j_route is None:
            continue
        if i_route is j_route:
            continue
        if not (D.nodes[i_id]["isLinkedToFinish"] and D.nodes[j_id]["isLinkedToStart"]):
            continue
        if i_route["cost"] + j_route["cost"] - ij_attr["savings"] > route_max_cost:
            continue
        ji = (j_id, i_id)
        if ji in efficiency_list:
            efficiency_list.remove(ji)
        merge_routes(i_id, j_id, ij_attr, D, sol)

    sol["routes"].sort(key=operator.itemgetter("demand"), reverse=True)
    if len(sol["routes"]) > fleet_size:
        for route in sol["routes"][fleet_size:]:
            sol["demand"] -= route["demand"]
            sol["cost"] -= route["cost"]
        sol["routes"] = sol["routes"][:fleet_size]
    else:
        sol["demand"] = sum(r["demand"] for r in sol["routes"])
    return sol

def plot_top_solution(D: nx.DiGraph, routes: list, instance_name: str, total_demand: float, total_cost: float):
    plt.figure(figsize=(12, 8))
    pos = {node: (D.nodes[node]["x"], D.nodes[node]["y"]) for node in D.nodes()}
    depot_nodes = [0, max(D.nodes())]
    customer_nodes = [node for node in D.nodes() if node not in depot_nodes]

    nx.draw_networkx_nodes(D, pos, nodelist = customer_nodes, node_size = 100, node_color = "lightblue", alpha = 0.7)
    nx.draw_networkx_nodes(D, pos, nodelist = depot_nodes, node_size = 200, node_color = "red", node_shape = "s")
    nx.draw_networkx_labels(D, pos, font_size = 8)
    colors = plt.cm.Set3.colors
    for i, route in enumerate(routes):
        route_color = colors[i % len(colors)]
        nx.draw_networkx_edges(D, pos, edgelist = route["edges"], edge_color = route_color, width = 2.5, alpha = 0.8)
    plt.title(f"PJS Heuristic Solution for {instance_name}\n"
              f"Total Reward: {total_demand:.2f}, Total Cost: {total_cost:.2f}")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(f"./unit-04/output/{instance_name}_PJS.png", dpi=300, bbox_inches="tight")

#######################################################################
###                         SOLVING & RESULTS                       ###
#######################################################################
if __name__ == "__main__":
    alpha = 0.7
    instance_name = "p2.2.f"
    fleet_size, route_max_cost, nodes = load_instance(instance_name)

    t0 = time.time()
    D_global = generate_graph(nodes)
    sol = pjs_solution(D_global, fleet_size, route_max_cost)
    t1 = time.time()

    # Result
    print(f"Instance: {instance_name}")
    print(f"Reward: {sol['demand']:.2f}")
    print(f"Computational time: {t1 - t0:.2f}sec.")
    for route in sol["routes"]:
        s = "0"
        for edge in route["edges"]:
            s += "-" + str(edge[1])
        print("Route:", s, "|| Reward =", round(route["demand"], 2), "|| Cost/Time =", round(route["cost"], 2))
    plot_top_solution(D_global, sol["routes"], instance_name, sol["demand"], sol["cost"])
