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
    Q_x = 1
    Q_y = 1
    Q_theta = 1
    R1 = 100
    R2 = 100

    step_horizon = 0.5
    N = 20

    # specs
    x_init = 20
    y_init = 20
    theta_init = 0

    v_max = 20
    v_min = -20
    theta_min = -2
    theta_max = 2

    # Control input constraints (linear speed and angular speed)
    x = ca.SX.sym('x')
    y = ca.SX.sym('y')
    theta = ca.SX.sym('theta')
    states = ca.vertcat(x, y, theta)
    n_states = states.numel()

    v = ca.SX.sym('v')
    w = ca.SX.sym('w')
    controls = ca.vertcat(v, w)
    n_controls = controls.numel()

    # matrix containing all states over all time steps +1 (each column is a state vector)
    X = ca.SX.sym('X', n_states, N + 1)
    # matrix containing all control actions over all time steps (each column is an action vector)
    U = ca.SX.sym('U', n_controls, N)
    # coloumn vector for storing initial state and target state
    P = ca.SX.sym('P', (N + 1) * n_states)

    # state weights matrix (Q_X, Q_Y, Q_THETA)
    Q = ca.diagcat(Q_x, Q_y, Q_theta)
    # controls weights matrix
    R = ca.diagcat(R1, R2)

    RHS = ca.vertcat(v * cos(theta), -v * sin(theta), w)
    # maps controls from [va, vb, vc, vd].T to [vx, vy, omega].T
    f = ca.Function('f', [states, controls], [RHS])
    accel_constraints = []
    cost_fn = 0  # cost function
    st = X[:, 0]
    lbg = ca.DM([])
    ubg = ca.DM([])
    g = st - P[0:3]  # constraints in the equation
    lbg = ca.vertcat(lbg, [0, 0, 0])
    ubg = ca.vertcat(ubg, [0, 0, 0])
    # runge kutta
    con_prev = U[0, 0]
    for k in range(N):
        st = X[:, k]
        con = U[:, k]
        cost_fn = cost_fn \
                  + (st - P[n_states * (k + 1): n_states + n_states * (k + 1)]).T @ Q @ (st - P[n_states * (k + 1): n_states + n_states * (k + 1)]) \
                  + con.T @ R @ con
        st_next = X[:, k + 1]
        # k1 = f(st, con)
        # k2 = f(st + step_horizon/2*k1, con)
        # k3 = f(st + step_horizon/2*k2, con)
        # k4 = f(st + step_horizon * k3, con)
        # st_next_RK4 = st + (step_horizon / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        # Acc constraints
        acc = (con[0] - con_prev) / step_horizon
        g = ca.vertcat(g, acc)  # constraint on velocity acceleration
        lbg = ca.vertcat(lbg, [-0.5])  # lower bound for velocity acceleration
        ubg = ca.vertcat(ubg, [0.5])  # upper bound for velocity acceleration
        con_prev = con[0]
        # Normal constraint
        f_value = f(st, con)
        st_next_euler = st + step_horizon * f_value
        g = ca.vertcat(g, st_next - st_next_euler)
        lbg = ca.vertcat(lbg, [0, 0, 0])
        ubg = ca.vertcat(ubg, [0, 0, 0])


    OPT_variables = ca.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1)))
    nlp_prob = {
        'f': cost_fn,
        'x': OPT_variables,
        'g': g,
        'p': P
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

    lbx = ca.DM.zeros((n_states * (N + 1) + n_controls * N, 1))
    ubx = ca.DM.zeros((n_states * (N + 1) + n_controls * N, 1))

    lbx[0: n_states * (N + 1): n_states] = -ca.inf  # X lower bound
    lbx[1: n_states * (N + 1): n_states] = -ca.inf  # Y lower bound
    lbx[2: n_states * (N + 1): n_states] = -ca.inf  # theta lower bound

    ubx[0: n_states * (N + 1): n_states] = ca.inf  # X upper bound
    ubx[1: n_states * (N + 1): n_states] = ca.inf  # Y upper bound
    ubx[2: n_states * (N + 1): n_states] = ca.inf  # theta upper bound

    lbx[n_states * (N + 1)::n_controls] = v_min  # v lower bound for all V
    ubx[n_states * (N + 1)::n_controls] = v_max  # v upper bound for all V

    lbx[1+n_states * (N + 1)::n_controls] = theta_min  # theta_min
    ubx[1+n_states * (N + 1)::n_controls] = theta_max  # theta_max



    args = {
        'lbg': lbg,  # constraints lower bound
        'ubg': ubg,
        'lbx': lbx,
        'ubx': ubx
    }

    t0 = 0
    state_init = ca.DM([x_init, y_init, theta_init])  # initial state

    t = ca.DM(t0)

    u0 = ca.DM.zeros((n_controls, N))  # initial control
    X0 = ca.repmat(state_init, 1, N + 1)  # initial state full

    mpc_iter = 0
    cat_states = DM2Arr(X0)
    cat_controls = DM2Arr(u0[:, 0])
    times = np.array([[0]])
    ii = 0
    x_sim = []
    c = []

    def __init__(self, dt, reference_trajectory):
        pygame.sprite.Sprite.__init__(self)
        self.dt = dt
        self.reference_trajectory = reference_trajectory
        self.state_target = ca.DM([reference_trajectory[0][0],
                                  reference_trajectory[0][1],
                                  reference_trajectory[0][2]])  # target state

    def move_and_rotate(self, screen):
        v, w = self.do_commands(screen)
    acc = []

    u_prev = 0
    def do_commands(self, screen):
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

        sol = self.solver(
            x0=self.args['x0'],
            lbx=self.args['lbx'],
            ubx=self.args['ubx'],
            lbg=self.args['lbg'],
            ubg=self.args['ubg'],
            p=self.args['p']
        )
        u = ca.reshape(sol['x'][self.n_states * (self.N + 1):], self.n_controls, self.N)
        X0 = ca.reshape(sol['x'][: self.n_states * (self.N + 1)], self.n_states, self.N + 1)

        if self.ii > 1:
         print((u[0, 0] - self.u_prev)/self.step_horizon)
        self.u_prev = u[0, 0]
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
        self.x_sim.append([x_optimized[0, 0], x_optimized[1, 0], x_optimized[2, 0]])
        #self.acc.append(DM2Arr(u0[0]))
        self.ii += 1
        return 0, 1