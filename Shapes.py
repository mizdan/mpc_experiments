import pygame
import numpy as np
import cvxpy as cp
import math

class Shape:
    def __init__(self, color, width, length):
        self.color = color
        self.width = width
        self.length = length

    def draw(self, screen, width):
        raise NotImplementedError("draw method must be implemented in subclass")

class Dock(Shape):
    dock_color = (255, 0, 0)
    dock_length = 50
    dock_width = 50

    def __init__(self):
        super().__init__(self.dock_color, self.dock_width, self.dock_length)

    def draw(self, screen, width):
        pygame.draw.rect(screen, self.dock_color, pygame.Rect(width / 2 - self.dock_length / 2, 0, self.dock_length, self.dock_width))
        pygame.draw.line(screen, self.dock_color, (width / 2, 0), (width / 2, 200))
        pass


class Player(pygame.sprite.Sprite):
    # Control input constraints (linear speed and angular speed)
    v_min, omega_min = -5, -0.5
    v_max, omega_max = 5, 0.5

    # Define cost matrices and prediction horizon
    Q = np.diag([100, 100, 0.1])  # State cost
    R = np.diag([1, 1])  # Control input cost
    N = 10  # Prediction horizon

    # Initial state and reference trajectory
    reference_trajectory = np.ones((1000, 3))
    for xx in range(0, 100):
        reference_trajectory[xx] = (300, 400 - xx, np.pi / 2)

    #for xx in range

    # Simulation parameters
    num_steps = 1000
    x_sim = np.zeros((num_steps + 1, 3))
    x_sim[0] = [288, 400, 0]
    u_sim = np.zeros((num_steps, 2))
    ii = 0

    def __init__(self, image, x, y):
        pygame.sprite.Sprite.__init__(self)
        self.original_image = image
        self.image = self.original_image
        self.rect = self.image.get_rect(center=(x, y))
        self.angle = 0
        self.x = x
        self.y = y

    def move_and_rotate(self, dt):
        v, w = self.do_commands()
        print("path = " + str(self.x) + " " + str(self.y))
        self.x = self.x + v * math.cos(self.angle) * dt
        self.y = self.y - v * math.sin(self.angle) * dt
        self.angle = self.angle + w * dt
        self.image = pygame.transform.rotate(self.original_image, -np.rad2deg(self.angle))

    def draw(self, screen):
        screen.blit(self.image, (self.x, self.y))
        self.rect = self.image.get_rect(center=self.rect.center)
        pygame.draw.line(screen, (255, 255, 255), self.reference_trajectory[0][0:2], self.reference_trajectory[99][0:2])

    def do_commands(self):
        A = np.array([[1.0, 0, 0],
                      [0, 1.0, 0],
                      [0, 0, 1.0]])

        C = np.eye(3)  # Assuming output equals state for simplicity
        # Define variables for optimization
        x = cp.Variable((self.N + 1, 3))
        u = cp.Variable((self.N, 2))

        B = np.array([[np.cos(self.x_sim[self.ii][2]), 0],
                      [-np.sin(self.x_sim[self.ii][2]), 0],
                      [0, 1]])
        # Define the cost function
        dy = self.reference_trajectory[0][1] - self.x_sim[self.ii][1]
        print(dy)
        cost = 0
        for i in range(self.N):
            cost += cp.quad_form(x[i] - self.reference_trajectory[int(dy + i)], self.Q) + cp.quad_form(u[i], self.R)

        # Define constraints
        constraints = [x[0] == self.x_sim[self.ii]]
        for i in range(self.N):
            constraints += [x[i + 1] == A @ x[i] + B @ u[i]]
            constraints += [u[i, 0] >= self.v_min, u[i, 0] <= self.v_max]  # Linear speed constraint
            constraints += [u[i, 1] >= self.omega_min, u[i, 1] <= self.omega_max]  # Angular speed constraint

        # Create and solve the optimization problem
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve()

        # Get the optimal control input
        optimal_u = u.value[0]
        # Simulate the system with the optimal control input
        self.x_sim[self.ii + 1] = A @ self.x_sim[self.ii] + B @ optimal_u
        self.u_sim[self.ii] = optimal_u
        self.ii += 1
        print("u_val = " + str(u.value[0]))
        return u.value[0]
