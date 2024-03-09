import pygame
import numpy as np

import math
import casadi as ca

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
    v_min, omega_min = -5, -2
    v_max, omega_max = 5, 2

    # Define cost matrices and prediction horizon
    Q = np.diag([1, 1, 1])  # State cost
    R = np.diag([1, 1])  # Control input cost
    N = 10  # Prediction horizon

    # Initial state and reference trajectory
    #reference_trajectory = np.ones((150, 3))
    ii = 0
    c = []
    opti = ca.Opti()

    def __init__(self, image, x, y, reference_trajectory):
        pygame.sprite.Sprite.__init__(self)
        self.original_image = image
        self.image = self.original_image
        self.rect = self.image.get_rect(center=(x, y))
        self.angle = 0
        self.x = x
        self.y = y
        self.reference_trajectory = reference_trajectory
        #self.x_sim = ca.SX.sym('x_sim', 3, (len(self.reference_trajectory) + 1))
        self.x_sim = np.zeros((len(self.reference_trajectory) + 1, 3))
        dx = reference_trajectory[0][0] - 2 - reference_trajectory[0][0]
        dy = reference_trajectory[0][1] + 2 - reference_trajectory[0][1]
        angle = math.atan2(dy, dx)
        self.x_sim[0:3] = [reference_trajectory[0][0]-2, reference_trajectory[0][1]+2, angle]
        self.u_sim = ca.SX.sym('u_sim', 2, len(self.reference_trajectory))

    def move_and_rotate(self, dt, screen):
        v, w = self.do_commands(screen)
        self.x = self.x + v * math.cos(self.angle) * dt
        self.y = self.y + v * math.sin(self.angle) * dt
        self.angle = self.angle + w * dt
        self.image = pygame.transform.rotate(self.original_image, np.rad2deg(self.angle + np.pi/2))
        self.rect = self.image.get_rect(center=(self.x, self.y))

    def draw(self, screen):
        screen.blit(self.image, self.rect)
        self.rect = self.image.get_rect(center=self.rect.center)
        for i in range(self.ii - 1):
            pygame.draw.line(screen, (0, 0, 0), self.x_sim[i][0:2], self.x_sim[i+1][0:2])

    def draw_prediction_horizon(self, screen, xi, i):
        pygame.draw.line(screen, (0, 0, 0), xi[i][0:2], xi[i + 1][0:2])

    def do_commands(self, screen):
        A = np.array([[1.0, 0, 0],
                      [0, 1.0, 0],
                      [0, 0, 1.0]])


        # Define symbolic variables for x and u
        x = self.opti.variable(3, self.N + 1)
        u = self.opti.variable(2, self.N)

        # Get theta and B
        B = np.array([[np.cos(self.x_sim[self.ii][2]), 0],
                      [-np.sin(self.x_sim[self.ii][2]), 0],
                      [0, 1]])

        min_distance = np.inf
        k_min = None
        current_distance = 0
        for idx in range(0, 5):
            if 0 <= self.ii + idx < len(self.x_sim) and self.ii + idx < len(self.reference_trajectory):
                dx = self.reference_trajectory[self.ii + idx][0] - self.x_sim[self.ii + idx][0]
                dy = self.reference_trajectory[self.ii + idx][1] - self.x_sim[self.ii + idx][1]
                current_distance = dx ** 2 + dy ** 2
            if current_distance < min_distance:
                min_distance = current_distance
                k_min = self.ii

        # Define cost
        cost = 0
        for i in range(self.N):
            dx = self.x - self.reference_trajectory[int(self.ii + i)][0]
            dy = self.y - self.reference_trajectory[int(self.ii + i)][1]
            cost += (x[:, i] -  self.reference_trajectory[int(k_min + i)]).T @ self.Q @ (
                        x[:, i] -  self.reference_trajectory[int(k_min + i)]) + u[:, i].T @ self.R @ u[:, i]

        # Define constraints
        self.opti.subject_to(x[:, 0] == self.x_sim[self.ii])

        for i in range(self.N):
            self.opti.subject_to(x[:, i + 1] == A @ x[:, i] + B @ u[:, i])
            self.opti.subject_to(u[0, i] >= self.v_min)
            self.opti.subject_to(u[0, i] <= self.v_max)
            self.opti.subject_to(u[1, i] >= self.omega_min)
            self.opti.subject_to(u[1, i] <= self.omega_max)


        # Set objective and solve
        self.opti.minimize(cost)
        self.opti.solver('ipopt')
        sol = self.opti.solve()

        # Get the optimal control input
        optimal_u = sol.value(u[:, 0])
        x_optimized = sol.value(x)

        for i in range(x_optimized.shape[1]):
            pygame.draw.circle(screen, (106, 90, 205), (int(x_optimized[0, i]), int(x_optimized[1, i])), 1)

        # Simulate the system with the optimal control input
        self.x_sim[self.ii + 1] = A @ self.x_sim[self.ii] + B @ optimal_u
        self.u_sim[:, self.ii] = optimal_u
        self.ii += 1
        return optimal_u[0], optimal_u[1]

