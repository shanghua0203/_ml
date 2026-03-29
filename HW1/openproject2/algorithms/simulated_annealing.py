import random
import math
from .core import calculate_route_distance, get_neighbor


class SimulatedAnnealingSolver:
    def __init__(self, cities, initial_temp=10000, cooling_rate=0.9999, min_temp=0.001):
        self.cities = cities
        self.n = len(cities)
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.temperature = initial_temp
        self.current_route = list(range(self.n))
        self.current_distance = calculate_route_distance(self.current_route, cities)
        self.best_route = self.current_route.copy()
        self.best_distance = self.current_distance
        self.iteration = 0
        self.total_iterations = 0
        self.last_new_route = None
        self.is_complete = False

    def step(self, iterations_per_step=100):
        self.last_new_route = None
        improved = False

        for _ in range(iterations_per_step):
            if self.temperature <= self.min_temp:
                self.is_complete = True
                break

            new_route = get_neighbor(self.current_route)
            new_distance = calculate_route_distance(new_route, self.cities)
            delta = new_distance - self.current_distance
            self.iteration += 1

            if delta < 0 or random.random() < math.exp(-delta / self.temperature):
                self.current_route = new_route
                self.current_distance = new_distance
                self.last_new_route = new_route

                if self.current_distance < self.best_distance:
                    self.best_route = self.current_route.copy()
                    self.best_distance = self.current_distance
                    improved = True

            self.temperature *= self.cooling_rate

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
            "temperature": self.temperature,
            "is_complete": self.is_complete,
        }

    def reset(self):
        self.temperature = self.initial_temp
        self.current_route = list(range(self.n))
        self.current_distance = calculate_route_distance(
            self.current_route, self.cities
        )
        self.best_route = self.current_route.copy()
        self.best_distance = self.current_distance
        self.iteration = 0
        self.total_iterations = 0
        self.last_new_route = None
        self.is_complete = False


def create_solver(cities):
    return SimulatedAnnealingSolver(cities)
