import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import math

# Define system matrices
A = np.array([[1.0, 0, 0],
              [0, 1.0, 0],
              [0, 0, 1.0]])



C = np.eye(3)  # Assuming output equals state for simplicity

# Define cost matrices and prediction horizon
Q = np.diag([1, 1, 1])  # State cost
R = np.diag([1, 1])       # Control input cost
N = 10                    # Prediction horizon

# Initial state and reference trajectory

reference_trajectory = np.ones((100, 3))
for xx in range(0, 40):
    reference_trajectory[xx] = (300, 400-xx, np.pi/2)


# Control input constraints (linear speed and angular speed)
v_min, omega_min = -5, -0.5
v_max, omega_max = 5, 0.5

# Simulation parameters
num_steps = 40
x_sim = np.zeros((num_steps + 1, 3))
x_sim[0] = [288, 400, 0]
u_sim = np.zeros((num_steps, 2))

# Define variables for optimization
x = cp.Variable((N+1, 3))
u = cp.Variable((N, 2))

for k in range(num_steps):
    B = np.array([[np.cos(x_sim[k][2]), 0],
                  [-np.sin(x_sim[k][2]), 0],
                  [0, 1]])
    # Define the cost function
    dy = reference_trajectory[0][1] - x_sim[k][1]
    print(dy)
    cost = 0
    for i in range(N):
        cost += cp.quad_form(x[i] - reference_trajectory[int(dy + i)], Q) + cp.quad_form(u[i], R)

    # Define constraints
    constraints = [x[0] == x_sim[k]]
    for i in range(N):
        constraints += [x[i + 1] == A @ x[i] + B @ u[i]]
        constraints += [u[i, 0] >= v_min, u[i, 0] <= v_max]          # Linear speed constraint
        constraints += [u[i, 1] >= omega_min, u[i, 1] <= omega_max]  # Angular speed constraint

    # Create and solve the optimization problem
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve()

    # Get the optimal control input
    optimal_u = u.value[0]
    # Simulate the system with the optimal control input
    x_sim[k + 1] = A @ x_sim[k] + B @ optimal_u
    u_sim[k] = optimal_u

# Plot results
plt.figure(figsize=(12, 8))

# Plot system output and reference trajectory
for i in range(3):
    plt.subplot(3, 1, i + 1)
    plt.plot(range(num_steps + 1), x_sim[:, i], label=f'State {i + 1} Output')
    plt.plot(range(100), reference_trajectory[:, i], linestyle='dashed', label=f'Reference Trajectory {i + 1}')
    plt.title(f'State {i + 1} Output and Reference Trajectory')
    plt.legend()

plt.tight_layout()
plt.show()
