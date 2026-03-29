import random
from .core import calculate_route_distance, get_neighbor, height


class HillClimbingSolver:
    def __init__(self, cities):
        self.cities = cities
        self.n = len(cities)
        self.current_route = list(range(self.n))
        self.current_distance = calculate_route_distance(self.current_route, cities)
        self.best_route = self.current_route.copy()
        self.best_distance = self.current_distance
        self.iteration = 0
        self.total_iterations = 0
        self.last_new_route = None

    def step(self, iterations_per_step=1000):
        self.last_new_route = None
        improved = False
        best_neighbor = None
        best_neighbor_distance = self.current_distance

        for _ in range(iterations_per_step):
            new_route = get_neighbor(self.current_route)
            new_distance = calculate_route_distance(new_route, self.cities)
            self.iteration += 1

            if new_distance < best_neighbor_distance:
                best_neighbor = new_route
                best_neighbor_distance = new_distance
                self.last_new_route = new_route

        if best_neighbor is not None and best_neighbor_distance < self.current_distance:
            self.current_route = best_neighbor
            self.current_distance = best_neighbor_distance
            improved = True

            if self.current_distance < self.best_distance:
                self.best_route = self.current_route.copy()
                self.best_distance = self.current_distance

        self.total_iterations += iterations_per_step
        return improved

    def get_state(self):
        return {
            "current_route": self.current_route.copy(),
            "current_distance": self.current_distance,
            "best_route": self.best_route.copy(),
            "best_distance": self.best_distance,
            "iteration": self.total_iterations,
            "new_route": self.last_new_route.copy() if self.last_new_route else None,
            "is_new_best": self.last_new_route is not None
            and calculate_route_distance(self.last_new_route, self.cities)
            < self.best_distance + 0.001,
        }

    def reset(self):
        self.current_route = list(range(self.n))
        self.current_distance = calculate_route_distance(
            self.current_route, self.cities
        )
        self.best_route = self.current_route.copy()
        self.best_distance = self.current_distance
        self.iteration = 0
        self.total_iterations = 0
        self.last_new_route = None


def create_solver(cities):
    return HillClimbingSolver(cities)
