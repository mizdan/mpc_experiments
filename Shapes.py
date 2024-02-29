import pygame
import math
import numpy as np
import osqp
import scipy as sp
from scipy import sparse
import matplotlib.pyplot as plt

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
    umin = np.array([-1, -0.02])
    umax = np.array([1, 0.02])
    xmin = np.array([-np.inf, -np.inf, -np.inf])
    xmax = np.array([np.inf, np.inf, np.inf])

    Q = sparse.diags([10000, 10, 1])
    R = sparse.diags([10, 10])

    #x0 = np.array([0, 0, np.pi / 2])
    #xr = np.array([100, 100, 0])
    ctr = []
    N = 10

    lin_vell = []            # input
    angular_vell = []        # input

    x_model = []            # output
    y_model = []            # output
    theta_model = [0]       # output
    i = 1

    def __init__(self, image, x, y, xr):
        pygame.sprite.Sprite.__init__(self)
        self.original_image = image
        self.image = self.original_image
        self.rect = self.image.get_rect(center=(x, y))
        self.angle = 0
        self.x = x
        self.y = y
        self.x0 = np.array([self.x, self.y, self.angle])
        self.xr = xr

    def move_and_rotate(self, dt):
        v, w = self.do_commands()
        self.x = self.x - v * math.cos(self.angle) * dt
        self.y = self.y - v * math.sin(self.angle) * dt
        print(self.x)
        self.angle = self.angle + w * dt
        #print(self.angle)
        self.image = pygame.transform.rotate(self.original_image, -np.rad2deg(self.angle))
        #self.image = pygame.transform.rotate(self.original_image, -30)

    def draw(self, screen):
        screen.blit(self.image, (self.x, self.y))
        self.rect = self.image.get_rect(center=self.rect.center)

    def do_commands(self):
        Ad = sparse.csc_matrix([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])

        Bd = sparse.csc_matrix([
            [np.cos(np.deg2rad(self.theta_model[self.i - 1])), 0],
            [np.sin(np.deg2rad(self.theta_model[self.i - 1])), 0],
            [0, 1]])

        [nx, nu] = Bd.shape

        # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
        # - quadratic objective
        P = sparse.block_diag([sparse.kron(sparse.eye(self.N), self.Q), self.Q,
                               sparse.kron(sparse.eye(self.N), self.R)], format='csc')

        # - linear objective
        q = np.hstack([np.kron(np.ones(self.N), -self.Q.dot(self.xr)), -self.Q.dot(self.xr), np.zeros(self.N * nu)])

        # - linear dynamics
        Ax = sparse.kron(sparse.eye(self.N + 1), -sparse.eye(nx)) + sparse.kron(sparse.eye(self.N + 1, k=-1), Ad)
        Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, self.N)), sparse.eye(self.N)]), Bd)
        Aeq = sparse.hstack([Ax, Bu])
        leq = np.hstack([-self.x0, np.zeros(self.N * nx)])
        ueq = leq

        # - input and state constraints
        Aineq = sparse.eye((self.N + 1) * nx + self.N * nu)
        lineq = np.hstack([np.kron(np.ones(self.N + 1), self.xmin), np.kron(np.ones(self.N), self.umin)])
        uineq = np.hstack([np.kron(np.ones(self.N + 1), self.xmax), np.kron(np.ones(self.N), self.umax)])

        # - OSQP constraints
        A = sparse.vstack([Aeq, Aineq], format='csc')
        l = np.hstack([leq, lineq])
        u = np.hstack([ueq, uineq])

        # Create an OSQP object
        prob = osqp.OSQP()

        # Setup workspace
        prob.setup(P, q, A, l, u, warm_start=True)

        # Solve
        res = prob.solve()

        # Check solver status
        if res.info.status != 'solved':
            raise ValueError('OSQP did not solve the problem!')

        ctrl = res.x[-self.N * nu: -(self.N - 1) * nu]
        for I in range(1, len(ctrl), 2):
            self.ctr.append(I)

        #plt.figure(3)
        #plt.grid(True)
        #plt.xlabel('Discrete time step (k)')
        #plt.suptitle('lin_vel')
        #plt.plot(player.lin_vell)
        #plt.show()
        x0 = Ad.dot(self.x0) + Bd.dot(ctrl)
        self.x0 = x0
        print(str(ctrl[0]) + " " + str(ctrl[1]))
        lin_vel = ctrl[0]
        angular_vel = ctrl[1]
        #temp_x = x0[0]
        #temp_y = x0[1]
        #temp_theta = x0[2]
        #self.x.append(x0[0])
        #self.y_model.append(x0[1])
        self.theta_model.append(self.x0[2])
        self.lin_vell.append(ctrl[0])
        self.angular_vell.append(ctrl[1])
        # Update initial state
        l[:nx] = -self.x0
        u[:nx] = -self.x0
        prob.update(l=l, u=u)
        self.i += 1
        return lin_vel, angular_vel
