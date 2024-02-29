import math

import osqp
import numpy as np
import scipy as sp
from scipy import sparse
import matplotlib.pyplot as plt



# Componentele vectorului de stare
# x[0] - x
# x[1] - y
# x[2] - theta

# Constrangeri comanda, stare
# SE COMPLETEAZA
umin = np.array([-np.inf])
umax = np.array([np.inf])
xmin = np.array([-np.inf, -np.inf, -np.inf])
xmax = np.array([np.inf, np.inf, np.inf])

# Matrice penalizare
# SE COMPLETEAZA
Q = sparse.diags([100, 1, 1])
R = sparse.diags([10])

# Starea initiala si de referinta
# SE COMPLETEAZA
# coord x , coord y , unghiul
x0 = np.array([0, 0, np.pi/2])

xr = np.array([100, 100, 0])

N = 10

lin_vel = []        #input
angular_vel = []    #input
x = []              #output
y = []              #output
theta = [np.pi/2]         #output

# Config sim and disturbance
nsim = 200          #simulare

# Simulate in closed loop
for i in range(nsim):
    Ad = sparse.csc_matrix([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])

    # Matricea B
    Bd = sparse.csc_matrix([
        [np.cos(np.deg2rad(theta[i-1]))],
        [np.sin(np.deg2rad(theta[i-1]))],
        [0]])

    [nx, nu] = Bd.shape

    # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
    # - quadratic objective
    P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), Q,
                           sparse.kron(sparse.eye(N), R)], format='csc')

    # - linear objective
    q = np.hstack([np.kron(np.ones(N), -Q.dot(xr)), -Q.dot(xr), np.zeros(N * nu)])

    # - linear dynamics
    Ax = sparse.kron(sparse.eye(N + 1), -sparse.eye(nx)) + sparse.kron(sparse.eye(N + 1, k=-1), Ad)
    Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd)
    Aeq = sparse.hstack([Ax, Bu])
    leq = np.hstack([-x0, np.zeros(N * nx)])
    ueq = leq

    # - input and state constraints
    Aineq = sparse.eye((N + 1) * nx + N * nu)
    lineq = np.hstack([np.kron(np.ones(N + 1), xmin), np.kron(np.ones(N), umin)])
    uineq = np.hstack([np.kron(np.ones(N + 1), xmax), np.kron(np.ones(N), umax)])

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

    # Apply first control input to the plant
    ctrl = res.x[-N*nu: -(N-1)*nu]
    x0 = Ad.dot(x0) + Bd.dot(ctrl)

    #save log
    lin_vel.append(ctrl[0])
    #angular_vel.append(ctrl[1])
    x.append(x0[0])
    y.append(x0[1])
    theta.append(x0[2])

    # Update initial state
    l[:nx] = -x0
    u[:nx] = -x0
    prob.update(l=l, u=u)


plt.figure(1)
plt.grid(True)
plt.xlabel('Discrete time step (k)')
plt.suptitle('lin_vel')
plt.plot(lin_vel)

plt.figure(2)
plt.grid(True)
plt.xlabel('Discrete time step (k)')
plt.suptitle('angular_vel')
plt.plot(angular_vel)

plt.figure(3)
plt.grid(True)
plt.xlabel('Discrete time step (k)')
plt.suptitle('x')
plt.plot(x)

plt.figure(4)
plt.grid(True)
plt.xlabel('Discrete time step (k)')
plt.suptitle('y')
plt.plot(y)

plt.figure(5)
plt.grid(True)
plt.xlabel('Discrete time step (k)')
plt.suptitle('theta')
plt.plot(theta)

plt.show()