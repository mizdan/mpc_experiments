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
    def __init__(self, image, x, y, reference_trajectory):
        pygame.sprite.Sprite.__init__(self)
        self.original_image = image
        self.image = self.original_image
        self.rect = self.image.get_rect(center=(x, y))
        self.angle = 0
        self.x = x
        self.y = y
        self.reference_trajectory = reference_trajectory
        self.x_sim = np.zeros((len(self.reference_trajectory) + 1, 3))
        dx = reference_trajectory[0][0] - 2 - reference_trajectory[0][0]
        dy = reference_trajectory[0][1] + 2 - reference_trajectory[0][1]
        angle = math.atan2(dy, dx)
        self.x_sim[0] = [reference_trajectory[0][0]-2, reference_trajectory[0][1]+2, angle]
        self.u_sim = np.zeros((len(self.reference_trajectory), 2))

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
        #pygame.draw.line(screen, (0, 0, 0), self.reference_trajectory[50][0:2], self.reference_trajectory[99][0:2])
        #pygame.draw.line(screen, (255, 255, 255), self.reference_trajectory[100][0:2], self.reference_trajectory[149][0:2])


    def draw_prediction_horizon(self, screen, xi, i):
        pygame.draw.line(screen, (0, 0, 0), xi[i][0:2], xi[i + 1][0:2])


    def do_commands(self, screen):
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


        cost = 0
        for i in range(self.N):
            dx = self.x - self.reference_trajectory[int(k_min + i)][0]
            dy = self.y - self.reference_trajectory[int(k_min + 1)][1]
            cost += cp.quad_form(x[i] - self.reference_trajectory[int(k_min + i)], self.Q) + cp.quad_form(u[i], self.R)

        # Define constraints
        constraints = [x[0] == self.x_sim[self.ii]]

        for i in range(self.N):
            constraints += [x[i + 1] == A @ x[i] + B @ u[i]]
            constraints += [u[i, 0] >= self.v_min, u[i, 0] <= self.v_max]  # Linear speed constraint
            constraints += [u[i, 1] >= self.omega_min, u[i, 1] <= self.omega_max]  # Angular speed constraint

        problem = cp.Problem(cp.Minimize(cost), constraints)
        self.c.append(problem.solve(solver=cp.OSQP))

        # Get the optimal control input
        optimal_u = u.value[0]
        for i in range(len(x.value)):
            pygame.draw.circle(screen,(106, 90, 205), (x.value[i][0], x.value[i][1]), 1)

        # Simulate the system with the optimal control input
        self.x_sim[self.ii + 1] = A @ self.x_sim[self.ii] + B @ optimal_u
        self.u_sim[self.ii] = optimal_u
        print("usim= " + str(self.u_sim[self.ii]))
        self.ii += 1
        print("u_val = " + str(u.value[0]))
        return u.value[0]
