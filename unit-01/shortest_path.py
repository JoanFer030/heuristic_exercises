# Graph definition - Ddjacency matrix with weights
graph = {
    "D": {}, 
    "C": {"D": 2},
    "B": {"C": 1, "D": 4},
    "A": {"B": 2, "C": 5}
}
memo = {}

def shortest(node, final):
    if node in memo:
        return memo[node]
    
    # If the current node is the final node
    if node == final:
        return 0
    
    # Calculate the shortest from all possible choices (recurrent function)
    costs = []
    for neighbor, cost in graph[node].items():
        total_cost = cost + shortest(neighbor, final)
        costs.append(total_cost)
    
    # Save and return
    min_cost = min(costs)
    memo[node] = min_cost
    return min_cost

# From node "A" to node "D"
print("Costo mínimo de A a D:", shortest("A", "D"))
for node in graph:
    print(f"shortest({node}) = {shortest(node, "D")}")
