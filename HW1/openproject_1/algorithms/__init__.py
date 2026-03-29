from .core import (
    calculate_distance,
    calculate_route_distance,
    height,
    generate_random_cities,
)
from .hill_climbing import HillClimbingSolver, create_solver as create_hc_solver
from .brute_force import BruteForceSolver, create_solver as create_bf_solver
from .simulated_annealing import (
    SimulatedAnnealingSolver,
    create_solver as create_sa_solver,
)

__all__ = [
    "calculate_distance",
    "calculate_route_distance",
    "height",
    "generate_random_cities",
    "HillClimbingSolver",
    "BruteForceSolver",
    "SimulatedAnnealingSolver",
]

ALGORITHMS = {
    "Hill Climbing": create_hc_solver,
    "Brute Force": create_bf_solver,
    "Simulated Annealing": create_sa_solver,
}


def create_solver(algorithm_name, cities):
    if algorithm_name in ALGORITHMS:
        return ALGORITHMS[algorithm_name](cities)
    return create_hc_solver(cities)
