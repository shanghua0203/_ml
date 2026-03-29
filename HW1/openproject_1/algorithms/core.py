import math


def calculate_distance(city1, city2):
    x1, y1 = city1
    x2, y2 = city2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_route_distance(route, cities):
    total_distance = 0
    n = len(route)
    for i in range(n):
        current_city = route[i]
        next_city = route[(i + 1) % n]
        total_distance += calculate_distance(cities[current_city], cities[next_city])
    return total_distance


def height(route, cities):
    return -calculate_route_distance(route, cities)


def get_neighbor(route):
    import random

    n = len(route)
    new_route = route.copy()
    i, j = sorted(random.sample(range(n), 2))
    new_route[i : j + 1] = reversed(new_route[i : j + 1])
    return new_route


def generate_random_cities(count, width, height, margin=50):
    import random

    cities = {}
    positions = set()
    for i in range(count):
        while True:
            x = random.randint(margin, width - margin)
            y = random.randint(margin, height - margin)
            pos = (x, y)
            if pos not in positions:
                positions.add(pos)
                cities[i] = pos
                break
    return cities
