from itertools import permutations
from .core import calculate_route_distance


class BruteForceSolver:
    def __init__(self, cities):
        self.cities = cities
        self.n = len(cities)
        self.best_route = None
        self.best_distance = float("inf")
        self.current_permutation = 0
        self.total_permutations = None
        self.permutations = None
        self.iteration = 0

    def step(self, iterations_per_step=1):
        if self.permutations is None:
            indices = list(range(self.n))
            from itertools import permutations as perm_gen

            self.permutations = list(perm_gen(indices))
            self.total_permutations = len(self.permutations)

        improved = False
        for _ in range(
            min(
                iterations_per_step,
                (self.total_permutations or 0) - self.current_permutation,
            )
        ):
            if self.current_permutation >= (self.total_permutations or 0):
                break

            perm = self.permutations[self.current_permutation]
            distance = calculate_route_distance(perm, self.cities)
            self.iteration += 1

            if distance < self.best_distance:
                self.best_route = list(perm)
                self.best_distance = distance
                improved = True

            self.current_permutation += 1

        return improved

    def get_state(self):
        return {
            "current_route": self.best_route.copy()
            if self.best_route
            else list(range(self.n)),
            "current_distance": self.best_distance,
            "best_route": self.best_route.copy()
            if self.best_route
            else list(range(self.n)),
            "best_distance": self.best_distance,
            "iteration": self.iteration,
            "new_route": None,
            "is_new_best": False,
            "progress": self.current_permutation / self.total_permutations
            if self.total_permutations
            else 0,
            "is_complete": self.current_permutation >= self.total_permutations
            if self.total_permutations
            else False,
        }

    def reset(self):
        self.best_route = None
        self.best_distance = float("inf")
        self.current_permutation = 0
        self.total_permutations = None
        self.permutations = None
        self.iteration = 0


def create_solver(cities):
    return BruteForceSolver(cities)
