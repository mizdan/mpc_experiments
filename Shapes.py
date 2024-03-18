import pygame
import numpy as np
import math
import casadi as ca
from casadi import sin, cos, pi


def DM2Arr(dm):
    return np.array(dm.full())


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
    # setting matrix_weights' variables
    def __init__(self, dt, reference_trajectory):
        pygame.sprite.Sprite.__init__(self)
        self.dt = dt
        self.reference_trajectory = reference_trajectory
        self.state_target = ca.DM([reference_trajectory[0][0],
                                   reference_trajectory[0][1],
                                   reference_trajectory[0][2]])
        self.ii = 0
        self.Q_x = 1000
        self.Q_y = 1000
        self.Q_theta = 100
        self.R1 = 100
        self.R2 = 1

        self.step_horizon = 0.1
        self.N = 20

        # specs
        self.x_init = 20
        self.y_init = 20
        self.theta_init = 0

        self.v_max = 20
        self.v_min = -20
        self.theta_max = 3.14
        self.theta_min = -3.14
        # Control input constraints (linear speed and angular speed)
        self.x = ca.SX.sym('x')
        self.y = ca.SX.sym('y')
        self.theta = ca.SX.sym('theta')
        self.states = ca.vertcat(self.x, self.y, self.theta)
        self.n_states = self.states.numel()

        self.v = ca.SX.sym('v')
        self.w = ca.SX.sym('w')
        self.controls = ca.vertcat(self.v, self.w)
        self.n_controls = self.controls.numel()

        self.X = ca.SX.sym('X', self.n_states, self.N + 1)
        self.U = ca.SX.sym('U', self.n_controls, self.N)
        self.P = ca.SX.sym('P', (self.N + 1) * self.n_states)

        self.Q = ca.diagcat(self.Q_x, self.Q_y, self.Q_theta)
        self.R = ca.diagcat(self.R1, self.R2)
        self.RHS = ca.vertcat(self.v * cos(self.theta), -self.v * sin(self.theta), self.w)
        # maps controls from [va, vb, vc, vd].T to [vx, vy, omega].T
        self.f = ca.Function('f', [self.states, self.controls], [self.RHS])
        accel_constraints = []
       # cost function
        self.args = {}
        self.state_init = ca.DM([self.x_init, self.y_init, self.theta_init])  # initial state
        self.u0 = ca.DM.zeros((self.n_controls, self.N))  # initial control
        self.X0 = ca.repmat(self.state_init, 1, self.N + 1)  # initial state full
        self.velocity = []
        # runge kutta
        con_prev = self.U[0, 0]

    def prediction_horizon(self):

        lbg = ca.DM([])
        ubg = ca.DM([])
        lbg = ca.vertcat(lbg, [0, 0, 0])
        ubg = ca.vertcat(ubg, [0, 0, 0])

        cost_fn = 0
        st = self.X[:, 0]
        g = st - self.P[0:3]  # constraints in the equation
        for k in range(self.N):
            st = self.X[:, k]
            con = self.U[:, k]
            cost_fn = cost_fn \
                      + (st - self.P[self.n_states * (k + 1): self.n_states + self.n_states * (k + 1)]).T @ self.Q @ \
                      (st - self.P[self.n_states * (k + 1): self.n_states + self.n_states * (k + 1)]) \
                      + con.T @ self.R @ con
            st_next = self.X[:, k + 1]
            # k1 = f(st, con)
            # k2 = f(st + step_horizon/2*k1, con)
            # k3 = f(st + step_horizon/2*k2, con)
            # k4 = f(st + step_horizon * k3, con)
            # st_next_RK4 = st + (step_horizon / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            # Acc constraints
            # acc = (con[0] - con_prev) / step_horizon
            # g = ca.vertcat(g,  (con[0] - con_prev) / step_horizon)  # constraint on velocity acceleration
            # lbg = ca.vertcat(lbg, 0)  # lower bound for velocity acceleration
            # ubg = ca.vertcat(ubg, 0)  # upper bound for velocity acceleration
            # con_prev = con[0]
            # Normal constraint
            if len(self.velocity) > 0:
                g = ca.vertcat(g,  (self.velocity[self.ii - 1] - con[0]) / self.step_horizon)  # constraint on velocity acceleration
                lbg = ca.vertcat(lbg, -0.5)  # lower bound for velocity acceleration
                ubg = ca.vertcat(ubg, 0.5)  # upper bound for velocity acceleration
            f_value = self.f(st, con)
            st_next_euler = st + self.step_horizon * f_value
            g = ca.vertcat(g, st_next - st_next_euler)
            lbg = ca.vertcat(lbg, [0, 0, 0])
            ubg = ca.vertcat(ubg, [0, 0, 0])

        OPT_variables = ca.vertcat(self.X.reshape((-1, 1)), self.U.reshape((-1, 1)))
        nlp_prob = {
            'f': cost_fn,
            'x': OPT_variables,
            'g': g,
            'p': self.P
        }

        opts = {
            'ipopt': {
                'max_iter': 2000,
                'print_level': 0,
                'acceptable_tol': 1e-8,
                'acceptable_obj_change_tol': 1e-6
            },
            'print_time': 0
        }

        solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

        lbx = ca.DM.zeros((self.n_states * (self.N + 1) + self.n_controls * self.N, 1))
        ubx = ca.DM.zeros((self.n_states * (self.N + 1) + self.n_controls * self.N, 1))

        lbx[0: self.n_states * (self.N + 1): self.n_states] = -ca.inf  # X lower bound
        lbx[1: self.n_states * (self.N + 1): self.n_states] = -ca.inf  # Y lower bound
        lbx[2: self.n_states * (self.N + 1): self.n_states] = -ca.inf  # theta lower bound

        ubx[0: self.n_states * (self.N + 1): self.n_states] = ca.inf  # X upper bound
        ubx[1: self.n_states * (self.N + 1): self.n_states] = ca.inf  # Y upper bound
        ubx[2: self.n_states * (self.N + 1): self.n_states] = ca.inf  # theta upper bound

        lbx[self.n_states * (self.N + 1)::self.n_controls] = self.v_min  # v lower bound for all V
        ubx[self.n_states * (self.N + 1)::self.n_controls] = self.v_max  # v upper bound for all V

        lbx[1+self.n_states * (self.N + 1)::self.n_controls] = self.theta_min  # theta_min
        ubx[1+self.n_states * (self.N + 1)::self.n_controls] = self.theta_max  # theta_max

        self.args = {
            'lbg': lbg,
            'ubg': ubg,
            'lbx': lbx,
            'ubx': ubx
        }

        t0 = 0
        return solver

    def move_and_rotate(self, screen):
        v, w = self.do_commands(screen)
    acc = []
    u_prev = 0
    con_prev = 0
    def do_commands(self, screen):
        solver = self.prediction_horizon()
        self.args['p'] = ca.vertcat(self.state_init)
        for i in range(self.N):
            self.args['p'] = ca.vertcat(self.args['p'], self.reference_trajectory[i + self.ii][0])
            self.args['p'] = ca.vertcat(self.args['p'], self.reference_trajectory[i + self.ii][1])
            self.args['p'] = ca.vertcat(self.args['p'], self.reference_trajectory[i + self.ii][2])
        # optimization variable current state
        self.args['x0'] = ca.vertcat(
            ca.reshape(self.X0, self.n_states * (self.N + 1), 1),
            ca.reshape(self.u0, self.n_controls * self.N, 1)
        )

        sol = solver(
            x0=self.args['x0'],
            lbx=self.args['lbx'],
            ubx=self.args['ubx'],
            lbg=self.args['lbg'],
            ubg=self.args['ubg'],
            p=self.args['p']
        )
        u = ca.reshape(sol['x'][self.n_states * (self.N + 1):], self.n_controls, self.N)
        X0 = ca.reshape(sol['x'][: self.n_states * (self.N + 1)], self.n_states, self.N + 1)
        self.velocity.append(u[0,0])
        if self.ii >= 0:
            print(f"u[0, 0] = {u[0, 0]}, con_prev = {self.con_prev}, acceleration = {(u[0, 0] - self.con_prev) / self.step_horizon}")
            self.con_prev = u[0, 0]

        f_value = self.f(self.state_init, u[:, 0])
        next_state = ca.DM.full(self.state_init + (self.step_horizon * f_value))
        self.state_init = next_state
        u0 = ca.horzcat(
            u[:, 1:],
            ca.reshape(u[:, -1], -1, 1)
        )
        uu0 = DM2Arr(u0)
        self.acc.append(uu0[0, 0])


        X0 = ca.horzcat(
            X0[:, 1:],
            ca.reshape(X0[:, -1], -1, 1)
        )
        x_optimized = DM2Arr(X0)

        for i in range(x_optimized.shape[1]):
            pygame.draw.circle(screen, (106, 90, 205), (int(x_optimized[0, i]), int(x_optimized[1, i])), 2)
        #print(X0)
        #self.x_sim.append([x_optimized[0, 0], x_optimized[1, 0], x_optimized[2, 0]])
        #self.acc.append(DM2Arr(u0[0]))
        self.ii += 1
        return 0, 1