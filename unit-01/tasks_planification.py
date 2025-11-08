import networkx as nx
import matplotlib.pyplot as plt


tasks = {
    "T1": {"T2": 3, "T3": 6, "T4": 7},
    "T2": {"T5": 4, "T6": 5},
    "T3": {"T6": 2},
    "T4": {"T6": 3, "T7": 4},
    "T5": {"T8": 6},
    "T6": {"T8": 2},
    "T7": {"T8": 3},
    "T8": {}
}
memo = {}
next_min = {}

def min_time(task, final):
    if task in memo:
        return memo[task]
    if task == final:
        memo[task] = 0
        return 0
    times = []
    for nxt, duration in tasks[task].items():
        total = duration + min_time(nxt, final)
        times.append((total, nxt))
    min_total, next_task = min(times)
    memo[task] = min_total
    next_min[task] = next_task
    return min_total

# Calculate times and stored them
min_time("T1", "T8") # From T1 to T8

path = ["T1"]
while path[-1] in next_min:
    path.append(next_min[path[-1]])

print("Minimum times from each task:")
for t in tasks:
    print(f"{t}: {memo[t]} days")
print("Minimum path:", " → ".join(path), f"(Total time = {memo["T1"]} days)")

# Plot the graph
G = nx.DiGraph()
for task, edges in tasks.items():
    for nxt, cost in edges.items():
        G.add_edge(task, nxt, weight=cost)

pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=1500)
nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")
nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, arrowstyle="-|>", width=2)
edge_labels = {(u, v): d["weight"] for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red", font_size=12)
plt.title(f"Graph", fontsize=14)
plt.axis("off")
plt.savefig("initial.png")

# Minimum path
path_edges = list(zip(path[:-1], path[1:]))
nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color="green", width=4, arrows=True, arrowstyle="-|>")

plt.title(f"Minimum path: {" → ".join(path)} (Total time = {memo["T1"]} days)", fontsize=14)
plt.axis("off")
plt.savefig("./unit-01/output/minimum_path.png")
