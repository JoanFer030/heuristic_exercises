import math
import random
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


class Rotation:
    NO_ROTATION = 0
    ALLOW_ROTATION = 1


@dataclass
class Item:
    """Rectangular item to be loaded into a vehicle."""
    id: int
    width: float
    height: float
    weight: float
    client_id: int

    def rotated(self) -> "Item":
        """Return the item rotated by 90 degrees."""
        return Item(self.id, self.height, self.width, self.weight, self.client_id)

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class Client:
    """Client with spatial location and a set of items."""
    id: int
    x: float
    y: float
    items: list[Item]

    @property
    def total_weight(self) -> float:
        return sum(item.weight for item in self.items)

    @property
    def total_area(self) -> float:
        return sum(item.area for item in self.items)


@dataclass
class Vehicle:
    """Vehicle capacity constraints."""
    id: int
    max_width: float
    max_height: float
    max_weight: float


@dataclass
class Instance:
    """Complete 2L-VRP instance."""
    depot: Client
    clients: list[Client]
    vehicle: Vehicle
    rotation: Rotation


class PackingSolution:
    def __init__(self):
        self.positions: dict[int, tuple[float, float, float, float, bool]] = {}
        self.used_height: float = 0.0
        self.feasible: bool = True


class ShelfPacking:
    @staticmethod
    def pack(items: list[Item], vehicle: Vehicle, rotation: Rotation) -> PackingSolution:
        solution = PackingSolution()
        items_sorted = sorted(items, key=lambda i: max(i.width, i.height), reverse=True)
        shelves: list[tuple[float, float, float]] = []
        current_y = 0.0
        for item in items_sorted:
            placed = False
            orientations = [item]
            if rotation == Rotation.ALLOW_ROTATION:
                orientations.append(item.rotated())
            for orient in orientations:
                best_shelf_idx = None
                best_remaining = float("inf")
                for idx, (y, h, rem_w) in enumerate(shelves):
                    if orient.height <= h and orient.width <= rem_w:
                        remaining = rem_w - orient.width
                        if remaining < best_remaining:
                            best_remaining = remaining
                            best_shelf_idx = idx
                if best_shelf_idx is not None:
                    y, h, rem_w = shelves[best_shelf_idx]
                    x_pos = vehicle.max_width - rem_w

                    solution.positions[item.id] = (
                        x_pos,
                        y,
                        orient.width,
                        orient.height,
                        orient.width != item.width,
                    )
                    shelves[best_shelf_idx] = (y, h, rem_w - orient.width)
                    placed = True
                    break
            if placed:
                continue
            
            best_orient = item
            if rotation == Rotation.ALLOW_ROTATION and item.height > item.width:
                best_orient = item.rotated()
            if (
                best_orient.width <= vehicle.max_width
                and current_y + best_orient.height <= vehicle.max_height
            ):
                solution.positions[item.id] = (
                    0.0,
                    current_y,
                    best_orient.width,
                    best_orient.height,
                    best_orient.width != item.width,
                )
                shelves.append(
                    (current_y, best_orient.height, vehicle.max_width - best_orient.width)
                )
                current_y += best_orient.height
                placed = True
            if not placed:
                solution.feasible = False
                return solution
        solution.used_height = current_y
        return solution



class InstanceGenerator:
    @staticmethod
    def generate(
        n_clients: int = 20,
        grid_size: int = 100,
        max_items_per_client: int = 3,
        vehicle_width: float = 10.0,
        vehicle_height: float = 8.0,
        vehicle_weight: float = 100.0,
        rotation: Rotation = Rotation.ALLOW_ROTATION,
        seed: int = None,
    ) -> Instance:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        depot = Client(0, grid_size / 2, grid_size / 2, [])
        clients: list[Client] = []
        item_id = 1
        for cid in range(1, n_clients + 1):
            x, y = random.uniform(0, grid_size), random.uniform(0, grid_size)
            items: list[Item] = []
            for _ in range(random.randint(1, max_items_per_client)):
                items.append(
                    Item(
                        id=item_id,
                        width=random.uniform(1.0, vehicle_width / 2),
                        height=random.uniform(1.0, vehicle_height / 2),
                        weight=random.uniform(5.0, 20.0),
                        client_id=cid,
                    )
                )
                item_id += 1
            clients.append(Client(cid, x, y, items))
        vehicle = Vehicle(1, vehicle_width, vehicle_height, vehicle_weight)
        return Instance(depot, clients, vehicle, rotation)


class ClarkWrightBRA:
    def __init__(self, instance: Instance, alpha: float = 0.3):
        self.instance = instance
        self.alpha = alpha
        self.routes: list[list[int]] = []
        self.client_to_route: dict[int, int] = {}
        self.savings: list[tuple[float, int, int]] = []

    @staticmethod
    def distance(c1: Client, c2: Client) -> float:
        return math.hypot(c1.x - c2.x, c1.y - c2.y)

    def compute_savings(self):
        depot = self.instance.depot
        savings = []
        for i in range(len(self.instance.clients)):
            for j in range(i + 1, len(self.instance.clients)):
                ci, cj = self.instance.clients[i], self.instance.clients[j]
                s = (
                    self.distance(depot, ci)
                    + self.distance(depot, cj)
                    - self.distance(ci, cj)
                )
                savings.append((s, i, j))
        self.savings = sorted(savings, reverse=True)

    def biased_choice(self, candidates):
        if not candidates:
            return None
        probs = [(1 - self.alpha) ** i for i in range(len(candidates))]
        total = sum(probs)
        r = random.random() * total
        acc = 0.0
        for cand, p in zip(candidates, probs):
            acc += p
            if acc >= r:
                return cand
        return candidates[-1]

    def route_feasible(self, route: list[int]) -> bool:
        clients = [self.instance.clients[i] for i in route]
        if sum(c.total_weight for c in clients) > self.instance.vehicle.max_weight:
            return False

        items = [item for c in clients for item in c.items]
        packing = ShelfPacking.pack(items, self.instance.vehicle, self.instance.rotation)
        return packing.feasible

    def solve(self) -> list[list[int]]:
        self.routes = [[i] for i in range(len(self.instance.clients))]
        self.client_to_route = {i: i for i in range(len(self.instance.clients))}
        self.compute_savings()
        savings_pool = self.savings.copy()
        while savings_pool:
            _, i, j = self.biased_choice(savings_pool)
            savings_pool = [s for s in savings_pool if s[1:] != (i, j)]
            ri, rj = self.client_to_route[i], self.client_to_route[j]
            if ri == rj:
                continue
            route_i, route_j = self.routes[ri], self.routes[rj]
            candidates = [
                route_i + route_j,
                route_i + list(reversed(route_j)),
                list(reversed(route_i)) + route_j,
                list(reversed(route_i)) + list(reversed(route_j)),
            ]
            merged = None
            for cand in candidates:
                if self.route_feasible(cand):
                    merged = cand
                    break
            if merged is None:
                continue
            self.routes.append(merged)
            new_idx = len(self.routes) - 1
            for c in merged:
                self.client_to_route[c] = new_idx
            for idx in sorted([ri, rj], reverse=True):
                del self.routes[idx]
                for k in self.client_to_route:
                    if self.client_to_route[k] > idx:
                        self.client_to_route[k] -= 1

        return [[0] + [i + 1 for i in r] + [0] for r in self.routes]



class TwoLVRPSolver:
    def __init__(self, instance: Instance):
        self.instance = instance

    def solve_bra(self, n_iter: int = 50, alpha: float = 0.3) -> dict:
        best = {"distance": float("inf")}
        for it in range(n_iter):
            cw = ClarkWrightBRA(self.instance, alpha)
            routes = cw.solve()
            dist = self.total_distance(routes)
            if dist < best["distance"]:
                best = {
                    "routes": routes,
                    "distance": dist,
                    "iteration": it,
                    "n_vehicles": len(routes),
                }
        return best
    def total_distance(self, routes: list[list[int]]) -> float:
        clients = [self.instance.depot] + self.instance.clients
        dist = 0.0
        for r in routes:
            for i in range(len(r) - 1):
                dist += ClarkWrightBRA.distance(clients[r[i]], clients[r[i + 1]])
        return dist


def plot_routes(instance: Instance, solution: dict, filename: str = "routes.png"):
    fig, ax = plt.subplots(figsize=(10, 8))
    # Depot
    ax.scatter(instance.depot.x, instance.depot.y, s=200, c="red", marker="s", label="Depot")
    # Clients
    for c in instance.clients:
        ax.scatter(c.x, c.y, s=80, c="blue")
        ax.text(c.x, c.y, f"C{c.id}", fontsize=9, ha="center", va="bottom")
    colors = plt.cm.tab10.colors
    nodes = [instance.depot] + instance.clients
    for i, route in enumerate(solution["routes"]):
        color = colors[i % len(colors)]
        for j in range(len(route) - 1):
            c1, c2 = nodes[route[j]], nodes[route[j + 1]]
            ax.plot([c1.x, c2.x], [c1.y, c2.y], color=color, linewidth=2)
        ax.plot([c1.x, c2.x], [c1.y, c2.y], color=color, linewidth=2, label = f"Route {i}")
    ax.set_title(
        f"2L-VRP: {solution['n_vehicles']} vehicles - Distance {solution['distance']:.2f}"
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_all_packings(instance: Instance, solution: dict, filename: str = "packing_all_vehicles.png"):
    """
    Create a single figure with one subplot per vehicle,
    showing the 2D packing of each route.
    """
    routes = solution["routes"]
    n_routes = len(routes)
    n_cols = min(3, n_routes)
    n_rows = (n_routes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_routes == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    all_clients = [instance.depot] + instance.clients
    for idx, route in enumerate(routes):
        ax = axes[idx]
        route_clients = [all_clients[i] for i in route if i != 0]
        items = [item for c in route_clients for item in c.items]
        packing = ShelfPacking.pack(items, instance.vehicle, instance.rotation)

        ax.add_patch(
            plt.Rectangle(
                (0, 0),
                instance.vehicle.max_width,
                instance.vehicle.max_height,
                fill=False,
                linewidth=2,
            )
        )
        for item_id, (x, y, w, h, _) in packing.positions.items():
            ax.add_patch(plt.Rectangle((x, y), w, h, edgecolor="black", alpha=0.7))
            ax.text(x + w / 2, y + h / 2, str(item_id), ha="center", va="center", fontsize=8)
        used_area = sum(w * h for (_, _, w, h, _) in packing.positions.values())
        total_area = instance.vehicle.max_width * instance.vehicle.max_height
        ax.set_title(
            f"Vehicle {idx + 1}\n"
            f"Items: {len(items)}, Area usage: {used_area / total_area:.1%}",
            fontsize=10,
        )
        ax.set_xlim(0, instance.vehicle.max_width)
        ax.set_ylim(0, instance.vehicle.max_height)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")
    plt.suptitle("2D packing per vehicle", fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


############################################
N_CLIENTS = 25
MAX_ITEMS = 3
WIDTH = 7.5
HEIGHT = 15.0
WEIGHT = 250.0

if __name__ == "__main__":
    import time
    instance = InstanceGenerator.generate(
        n_clients=N_CLIENTS,
        max_items_per_client=MAX_ITEMS,
        vehicle_width=WIDTH,
        vehicle_height=HEIGHT,
        vehicle_weight=WEIGHT,
        seed=42,
    )

    t0 = time.time()
    solver = TwoLVRPSolver(instance)
    solution = solver.solve_bra(n_iter=50, alpha=0.3)

    print(f"Vehicles: {solution['n_vehicles']}")
    print(f"Total distance: {solution['distance']:.2f}")
    print(f"Total time: {time.time() - t0:.4f}s")

    plot_routes(instance, solution, f"./unit-06/output/C{N_CLIENTS}I{MAX_ITEMS}-{WIDTH}x{HEIGHT}_routes_BR-CWS.jpg")
    plot_all_packings(instance, solution, f"./unit-06/output/C{N_CLIENTS}I{MAX_ITEMS}-{WIDTH}x{HEIGHT}_packing_BR-CWS.jpg")