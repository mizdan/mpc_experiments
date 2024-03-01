import pygame
import numpy as np

pygame.init()
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Random Path")
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

def generate_random_path(num_points=4):
    def generate_random_point(min_distance=100):
        while True:
            point = np.random.rand(2) * np.array([width, height])
            if all(np.linalg.norm(point - p) > min_distance for p in points):
                return point

    points = []
    min_distance = 100
    first_point = np.random.rand(2) * np.array([width, height])
    points.append(first_point)
    while len(points) < num_points:
        new_point = generate_random_point(min_distance)
        points.append(new_point)
    points.append(first_point)
    return np.array(points)

def divide_path(path, num_divisions=10):
    divided_points = []
    for i in range(len(path) - 1):
        x_vals = np.linspace(path[i][0], path[i + 1][0], num_divisions)
        y_vals = np.linspace(path[i][1], path[i + 1][1], num_divisions)
        divided_points.extend(np.column_stack((x_vals, y_vals)))
    return divided_points

random_path = generate_random_path()
divided_points = divide_path(random_path)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(BLACK)
    pygame.draw.lines(screen, WHITE, True, random_path, 2)
    for i in range(len(divided_points) - 1):
        dx = divided_points[i][0] - divided_points[i + 1][0]
        dy = divided_points[i][1] - divided_points[i + 1][1]

    pygame.display.flip()

pygame.quit()
